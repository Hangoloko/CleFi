use anchor_lang::prelude::*;
use anchor_spl::token::{self, Burn, Mint, Token, TokenAccount, Transfer};

declare_id!("Fg6PaFpoGXkYsidMpWTK6W2BeZ7FEfcYkgP8i6J3vHcJ");

const BPS_DENOMINATOR: u64 = 10_000;
const REINVESTMENT_BPS: u64 = 500; // 5%
const MAX_PAYOUT: u64 = 1_000_000;
const RATE_LIMIT_SECONDS: i64 = 3600;
const MIN_RESERVE_RATIO_BPS: u64 = 2_500; // 25%
const PROPOSAL_DURATION_SECONDS: i64 = 7 * 24 * 3600;
const EXECUTION_DELAY_SECONDS: i64 = 2 * 24 * 3600;
const MIN_QUORUM: u64 = 100_000;
const PROPOSAL_THRESHOLD: u64 = 50_000;

#[program]
pub mod sustainable_defi_anchor {
    use super::*;

    pub fn initialize(
        ctx: Context<Initialize>,
        merkle_root: [u8; 32],
        initial_ai_params: AIControlledParameters,
    ) -> Result<()> {
        validate_ai_params(&initial_ai_params)?;

        let state = &mut ctx.accounts.state;
        let clock = Clock::get()?;

        state.admin = ctx.accounts.admin.key();
        state.mint = ctx.accounts.mint.key();
        state.merkle_root = merkle_root;
        state.deposit_vault = ctx.accounts.deposit_vault.key();
        state.reserve_vault = ctx.accounts.reserve_vault.key();
        state.fee_vault = ctx.accounts.fee_vault.key();
        state.vault_authority_bump = ctx.bumps.vault_authority;
        state.security_reserve_amount = 0;
        state.reward_pool_amount = 0;
        state.total_deposits = 0;
        state.proposal_counter = 0;
        state.last_optimization_timestamp = clock.unix_timestamp;
        state.is_paused = false;
        state.ai_params = initial_ai_params;

        Ok(())
    }

    pub fn initialize_user(ctx: Context<InitializeUser>) -> Result<()> {
        let user = &mut ctx.accounts.user_position;

        user.owner = ctx.accounts.user_authority.key();
        user.balance = 0;
        user.loyalty_points = 0;
        user.last_deposit = 0;
        user.last_operation_timestamp = 0;
        user.operations_in_window = 0;
        user.voting_delegate = None;

        Ok(())
    }

    pub fn deposit(ctx: Context<Deposit>, amount: u64) -> Result<()> {
        require!(amount > 0, ErrorCode::InvalidAmount);

        let clock = Clock::get()?;
        let state = &mut ctx.accounts.state;
        let user = &mut ctx.accounts.user_position;

        require!(!state.is_paused, ErrorCode::ProtocolPaused);
        require_keys_eq!(user.owner, ctx.accounts.user_authority.key(), ErrorCode::Unauthorized);

        if user.last_operation_timestamp != 0 {
            require!(
                clock.unix_timestamp - user.last_operation_timestamp >= RATE_LIMIT_SECONDS,
                ErrorCode::RateLimitExceeded
            );
        }

        let fee = amount
            .checked_mul(state.ai_params.adaptive_entry_fee_bps as u64)
            .ok_or(ErrorCode::MathOverflow)?
            .checked_div(BPS_DENOMINATOR)
            .ok_or(ErrorCode::MathOverflow)?;

        let burn_amount = amount
            .checked_mul(state.ai_params.dynamic_burn_bps as u64)
            .ok_or(ErrorCode::MathOverflow)?
            .checked_div(BPS_DENOMINATOR)
            .ok_or(ErrorCode::MathOverflow)?;

        let reinvestment = amount
            .checked_mul(REINVESTMENT_BPS)
            .ok_or(ErrorCode::MathOverflow)?
            .checked_div(BPS_DENOMINATOR)
            .ok_or(ErrorCode::MathOverflow)?;

        let allocated = fee
            .checked_add(burn_amount)
            .ok_or(ErrorCode::MathOverflow)?
            .checked_add(reinvestment)
            .ok_or(ErrorCode::MathOverflow)?;

        require!(amount > allocated, ErrorCode::DepositTooLow);

        let net_deposit = amount
            .checked_sub(allocated)
            .ok_or(ErrorCode::MathOverflow)?;

        token::transfer(ctx.accounts.transfer_to_deposit_vault_ctx(), net_deposit)?;
        token::transfer(ctx.accounts.transfer_to_reserve_vault_ctx(), reinvestment)?;
        token::transfer(ctx.accounts.transfer_to_fee_vault_ctx(), fee)?;
        token::burn(ctx.accounts.burn_ctx(), burn_amount)?;

        user.balance = user
            .balance
            .checked_add(net_deposit)
            .ok_or(ErrorCode::MathOverflow)?;
        user.loyalty_points = user
            .loyalty_points
            .checked_add(net_deposit / 1_000_000)
            .ok_or(ErrorCode::MathOverflow)?;
        user.last_deposit = clock.unix_timestamp;
        user.last_operation_timestamp = clock.unix_timestamp;
        user.operations_in_window = user.operations_in_window.saturating_add(1);

        state.security_reserve_amount = state
            .security_reserve_amount
            .checked_add(reinvestment)
            .ok_or(ErrorCode::MathOverflow)?;
        state.reward_pool_amount = state
            .reward_pool_amount
            .checked_add(fee)
            .ok_or(ErrorCode::MathOverflow)?;
        state.total_deposits = state
            .total_deposits
            .checked_add(net_deposit)
            .ok_or(ErrorCode::MathOverflow)?;

        optimize_parameters(state, amount, clock.unix_timestamp)?;

        emit!(DepositEvent {
            user: ctx.accounts.user_authority.key(),
            gross_amount: amount,
            net_amount: net_deposit,
            fee,
            burn_amount,
            reinvestment,
        });

        Ok(())
    }

    pub fn withdraw(ctx: Context<Withdraw>, amount: u64) -> Result<()> {
        require!(amount > 0, ErrorCode::InvalidAmount);
        require!(amount <= MAX_PAYOUT, ErrorCode::ExceedsMaxPayout);

        let state = &mut ctx.accounts.state;
        let user = &mut ctx.accounts.user_position;

        require!(!state.is_paused, ErrorCode::ProtocolPaused);
        require_keys_eq!(user.owner, ctx.accounts.user_authority.key(), ErrorCode::Unauthorized);
        require!(user.balance >= amount, ErrorCode::InsufficientBalance);

        let remaining_total = state
            .total_deposits
            .checked_sub(amount)
            .ok_or(ErrorCode::MathOverflow)?;

        if remaining_total > 0 {
            let reserve_ratio_bps = state
                .security_reserve_amount
                .checked_mul(BPS_DENOMINATOR)
                .ok_or(ErrorCode::MathOverflow)?
                .checked_div(remaining_total)
                .ok_or(ErrorCode::MathOverflow)?;
            require!(
                reserve_ratio_bps >= MIN_RESERVE_RATIO_BPS,
                ErrorCode::InsecureReserveRatio
            );
        }

        user.balance = user
            .balance
            .checked_sub(amount)
            .ok_or(ErrorCode::MathOverflow)?;
        state.total_deposits = remaining_total;

        let signer_seeds: &[&[u8]] = &[b"vault-authority", &[state.vault_authority_bump]];
        token::transfer(
            ctx.accounts
                .transfer_from_deposit_vault_ctx()
                .with_signer(&[signer_seeds]),
            amount,
        )?;

        emit!(WithdrawEvent {
            user: ctx.accounts.user_authority.key(),
            amount,
        });

        Ok(())
    }

    pub fn create_proposal(
        ctx: Context<CreateProposal>,
        action: ProposalAction,
    ) -> Result<()> {
        let state = &mut ctx.accounts.state;
        let proposer = &ctx.accounts.proposer_position;
        let clock = Clock::get()?;

        require_keys_eq!(
            proposer.owner,
            ctx.accounts.proposer_authority.key(),
            ErrorCode::Unauthorized
        );
        require!(proposer.balance >= PROPOSAL_THRESHOLD, ErrorCode::InsufficientProposalThreshold);

        if let ProposalAction::UpdateAIParameters(params) = &action {
            validate_ai_params(params)?;
        }

        let proposal = &mut ctx.accounts.proposal;
        proposal.id = state.proposal_counter;
        proposal.proposer = ctx.accounts.proposer_authority.key();
        proposal.action = action;
        proposal.status = ProposalStatus::Active;
        proposal.start_time = clock.unix_timestamp;
        proposal.end_time = clock.unix_timestamp + PROPOSAL_DURATION_SECONDS;
        proposal.votes_for = 0;
        proposal.votes_against = 0;
        proposal.executed = false;

        state.proposal_counter = state
            .proposal_counter
            .checked_add(1)
            .ok_or(ErrorCode::MathOverflow)?;

        emit!(ProposalCreatedEvent {
            proposal_id: proposal.id,
            proposer: proposal.proposer,
        });

        Ok(())
    }

    pub fn cast_vote(ctx: Context<CastVote>, support: bool) -> Result<()> {
        let proposal = &mut ctx.accounts.proposal;
        let voter_position = &ctx.accounts.voter_position;
        let clock = Clock::get()?;

        require_keys_eq!(
            voter_position.owner,
            ctx.accounts.voter_authority.key(),
            ErrorCode::Unauthorized
        );
        require!(proposal.status == ProposalStatus::Active, ErrorCode::ProposalNotActive);
        require!(
            clock.unix_timestamp >= proposal.start_time && clock.unix_timestamp <= proposal.end_time,
            ErrorCode::VotingPeriodInvalid
        );

        let voting_power = calculate_voting_power(voter_position, clock.unix_timestamp)?;

        let receipt = &mut ctx.accounts.vote_receipt;
        receipt.proposal = proposal.key();
        receipt.voter = ctx.accounts.voter_authority.key();
        receipt.support = support;
        receipt.voting_power = voting_power;

        if support {
            proposal.votes_for = proposal
                .votes_for
                .checked_add(voting_power)
                .ok_or(ErrorCode::MathOverflow)?;
        } else {
            proposal.votes_against = proposal
                .votes_against
                .checked_add(voting_power)
                .ok_or(ErrorCode::MathOverflow)?;
        }

        emit!(VoteCastEvent {
            proposal_id: proposal.id,
            voter: ctx.accounts.voter_authority.key(),
            support,
            voting_power,
        });

        Ok(())
    }

    pub fn finalize_proposal(ctx: Context<FinalizeProposal>) -> Result<()> {
        let proposal = &mut ctx.accounts.proposal;
        let clock = Clock::get()?;

        require!(proposal.status == ProposalStatus::Active, ErrorCode::ProposalNotActive);
        require!(clock.unix_timestamp > proposal.end_time, ErrorCode::VotingPeriodNotEnded);

        let total_votes = proposal
            .votes_for
            .checked_add(proposal.votes_against)
            .ok_or(ErrorCode::MathOverflow)?;

        let quorum_reached = total_votes >= MIN_QUORUM;

        if quorum_reached && proposal.votes_for > proposal.votes_against {
            proposal.status = ProposalStatus::Succeeded;
        } else {
            proposal.status = ProposalStatus::Failed;
        }

        emit!(ProposalFinalizedEvent {
            proposal_id: proposal.id,
            total_votes,
            quorum_reached,
            succeeded: proposal.status == ProposalStatus::Succeeded,
        });

        Ok(())
    }

    pub fn execute_proposal(ctx: Context<ExecuteProposal>) -> Result<()> {
        let state = &mut ctx.accounts.state;
        let proposal = &mut ctx.accounts.proposal;
        let clock = Clock::get()?;

        require!(proposal.status == ProposalStatus::Succeeded, ErrorCode::ProposalNotSucceeded);
        require!(!proposal.executed, ErrorCode::ProposalAlreadyExecuted);
        require!(
            clock.unix_timestamp >= proposal.end_time + EXECUTION_DELAY_SECONDS,
            ErrorCode::TimelockNotExpired
        );

        match &proposal.action {
            ProposalAction::PauseProtocol => {
                state.is_paused = true;
            }
            ProposalAction::ResumeProtocol => {
                state.is_paused = false;
            }
            ProposalAction::UpdateAIParameters(params) => {
                validate_ai_params(params)?;
                state.ai_params = params.clone();
            }
        }

        proposal.executed = true;
        proposal.status = ProposalStatus::Executed;

        emit!(ProposalExecutedEvent {
            proposal_id: proposal.id,
            executor: ctx.accounts.executor.key(),
        });

        Ok(())
    }

    pub fn set_delegate(ctx: Context<SetDelegate>, delegate: Option<Pubkey>) -> Result<()> {
        let user = &mut ctx.accounts.user_position;
        require_keys_eq!(user.owner, ctx.accounts.user_authority.key(), ErrorCode::Unauthorized);

        if let Some(d) = delegate {
            require!(d != ctx.accounts.user_authority.key(), ErrorCode::CircularDelegation);
        }

        user.voting_delegate = delegate;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(
        init,
        payer = admin,
        space = State::LEN,
        seeds = [b"state"],
        bump
    )]
    pub state: Account<'info, State>,

    #[account(
        seeds = [b"vault-authority"],
        bump
    )]
    /// CHECK: PDA used only as token authority.
    pub vault_authority: UncheckedAccount<'info>,

    #[account(
        init,
        payer = admin,
        token::mint = mint,
        token::authority = vault_authority
    )]
    pub deposit_vault: Account<'info, TokenAccount>,

    #[account(
        init,
        payer = admin,
        token::mint = mint,
        token::authority = vault_authority
    )]
    pub reserve_vault: Account<'info, TokenAccount>,

    #[account(
        init,
        payer = admin,
        token::mint = mint,
        token::authority = vault_authority
    )]
    pub fee_vault: Account<'info, TokenAccount>,

    pub mint: Account<'info, Mint>,

    #[account(mut)]
    pub admin: Signer<'info>,

    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct InitializeUser<'info> {
    #[account(
        init,
        payer = user_authority,
        space = UserPosition::LEN,
        seeds = [b"user-position", user_authority.key().as_ref()],
        bump
    )]
    pub user_position: Account<'info, UserPosition>,

    #[account(mut)]
    pub user_authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct Deposit<'info> {
    #[account(mut, seeds = [b"state"], bump)]
    pub state: Account<'info, State>,

    #[account(
        mut,
        seeds = [b"user-position", user_authority.key().as_ref()],
        bump,
        has_one = owner @ ErrorCode::Unauthorized
    )]
    pub user_position: Account<'info, UserPosition>,

    #[account(mut)]
    pub user_authority: Signer<'info>,

    #[account(
        mut,
        constraint = user_token_account.owner == user_authority.key() @ ErrorCode::Unauthorized,
        constraint = user_token_account.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub user_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        address = state.deposit_vault,
        constraint = deposit_vault.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub deposit_vault: Account<'info, TokenAccount>,

    #[account(
        mut,
        address = state.reserve_vault,
        constraint = reserve_vault.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub reserve_vault: Account<'info, TokenAccount>,

    #[account(
        mut,
        address = state.fee_vault,
        constraint = fee_vault.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub fee_vault: Account<'info, TokenAccount>,

    #[account(address = state.mint)]
    pub mint: Account<'info, Mint>,

    pub token_program: Program<'info, Token>,
}

impl<'info> Deposit<'info> {
    fn transfer_to_deposit_vault_ctx(&self) -> CpiContext<'_, '_, '_, 'info, Transfer<'info>> {
        CpiContext::new(
            self.token_program.to_account_info(),
            Transfer {
                from: self.user_token_account.to_account_info(),
                to: self.deposit_vault.to_account_info(),
                authority: self.user_authority.to_account_info(),
            },
        )
    }

    fn transfer_to_reserve_vault_ctx(&self) -> CpiContext<'_, '_, '_, 'info, Transfer<'info>> {
        CpiContext::new(
            self.token_program.to_account_info(),
            Transfer {
                from: self.user_token_account.to_account_info(),
                to: self.reserve_vault.to_account_info(),
                authority: self.user_authority.to_account_info(),
            },
        )
    }

    fn transfer_to_fee_vault_ctx(&self) -> CpiContext<'_, '_, '_, 'info, Transfer<'info>> {
        CpiContext::new(
            self.token_program.to_account_info(),
            Transfer {
                from: self.user_token_account.to_account_info(),
                to: self.fee_vault.to_account_info(),
                authority: self.user_authority.to_account_info(),
            },
        )
    }

    fn burn_ctx(&self) -> CpiContext<'_, '_, '_, 'info, Burn<'info>> {
        CpiContext::new(
            self.token_program.to_account_info(),
            Burn {
                mint: self.mint.to_account_info(),
                from: self.user_token_account.to_account_info(),
                authority: self.user_authority.to_account_info(),
            },
        )
    }
}

#[derive(Accounts)]
pub struct Withdraw<'info> {
    #[account(mut, seeds = [b"state"], bump)]
    pub state: Account<'info, State>,

    #[account(
        mut,
        seeds = [b"user-position", user_authority.key().as_ref()],
        bump,
        has_one = owner @ ErrorCode::Unauthorized
    )]
    pub user_position: Account<'info, UserPosition>,

    #[account(mut)]
    pub user_authority: Signer<'info>,

    #[account(
        mut,
        constraint = user_token_account.owner == user_authority.key() @ ErrorCode::Unauthorized,
        constraint = user_token_account.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub user_token_account: Account<'info, TokenAccount>,

    #[account(
        mut,
        address = state.deposit_vault,
        constraint = deposit_vault.mint == state.mint @ ErrorCode::InvalidMint
    )]
    pub deposit_vault: Account<'info, TokenAccount>,

    #[account(
        seeds = [b"vault-authority"],
        bump = state.vault_authority_bump
    )]
    /// CHECK: PDA used only as token authority.
    pub vault_authority: UncheckedAccount<'info>,

    pub token_program: Program<'info, Token>,
}

impl<'info> Withdraw<'info> {
    fn transfer_from_deposit_vault_ctx(
        &self,
    ) -> CpiContext<'_, '_, '_, 'info, Transfer<'info>> {
        CpiContext::new(
            self.token_program.to_account_info(),
            Transfer {
                from: self.deposit_vault.to_account_info(),
                to: self.user_token_account.to_account_info(),
                authority: self.vault_authority.to_account_info(),
            },
        )
    }
}

#[derive(Accounts)]
pub struct CreateProposal<'info> {
    #[account(mut, seeds = [b"state"], bump)]
    pub state: Account<'info, State>,

    #[account(
        seeds = [b"user-position", proposer_authority.key().as_ref()],
        bump,
        has_one = owner @ ErrorCode::Unauthorized
    )]
    pub proposer_position: Account<'info, UserPosition>,

    #[account(
        init,
        payer = proposer_authority,
        space = Proposal::LEN,
        seeds = [b"proposal", state.proposal_counter.to_le_bytes().as_ref()],
        bump
    )]
    pub proposal: Account<'info, Proposal>,

    #[account(mut)]
    pub proposer_authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CastVote<'info> {
    #[account(seeds = [b"state"], bump)]
    pub state: Account<'info, State>,

    #[account(mut)]
    pub proposal: Account<'info, Proposal>,

    #[account(
        seeds = [b"user-position", voter_authority.key().as_ref()],
        bump,
        has_one = owner @ ErrorCode::Unauthorized
    )]
    pub voter_position: Account<'info, UserPosition>,

    #[account(
        init,
        payer = voter_authority,
        space = VoteReceipt::LEN,
        seeds = [b"vote-receipt", proposal.key().as_ref(), voter_authority.key().as_ref()],
        bump
    )]
    pub vote_receipt: Account<'info, VoteReceipt>,

    #[account(mut)]
    pub voter_authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FinalizeProposal<'info> {
    pub state: Account<'info, State>,
    #[account(mut)]
    pub proposal: Account<'info, Proposal>,
}

#[derive(Accounts)]
pub struct ExecuteProposal<'info> {
    #[account(mut, seeds = [b"state"], bump)]
    pub state: Account<'info, State>,
    #[account(mut)]
    pub proposal: Account<'info, Proposal>,
    pub executor: Signer<'info>,
}

#[derive(Accounts)]
pub struct SetDelegate<'info> {
    #[account(
        mut,
        seeds = [b"user-position", user_authority.key().as_ref()],
        bump,
        has_one = owner @ ErrorCode::Unauthorized
    )]
    pub user_position: Account<'info, UserPosition>,
    pub user_authority: Signer<'info>,
}

#[account]
pub struct State {
    pub admin: Pubkey,
    pub mint: Pubkey,
    pub merkle_root: [u8; 32],
    pub deposit_vault: Pubkey,
    pub reserve_vault: Pubkey,
    pub fee_vault: Pubkey,
    pub vault_authority_bump: u8,
    pub security_reserve_amount: u64,
    pub reward_pool_amount: u64,
    pub total_deposits: u64,
    pub proposal_counter: u64,
    pub last_optimization_timestamp: i64,
    pub is_paused: bool,
    pub ai_params: AIControlledParameters,
}

impl State {
    pub const LEN: usize = 8 + 32 + 32 + 32 + 32 + 32 + 32 + 1 + 8 + 8 + 8 + 8 + 8 + 1 + AIControlledParameters::LEN;
}

#[account]
pub struct UserPosition {
    pub owner: Pubkey,
    pub balance: u64,
    pub loyalty_points: u64,
    pub last_deposit: i64,
    pub last_operation_timestamp: i64,
    pub operations_in_window: u8,
    pub voting_delegate: Option<Pubkey>,
}

impl UserPosition {
    pub const LEN: usize = 8 + 32 + 8 + 8 + 8 + 8 + 1 + (1 + 32);
}

#[account]
pub struct Proposal {
    pub id: u64,
    pub proposer: Pubkey,
    pub action: ProposalAction,
    pub status: ProposalStatus,
    pub start_time: i64,
    pub end_time: i64,
    pub votes_for: u64,
    pub votes_against: u64,
    pub executed: bool,
}

impl Proposal {
    pub const LEN: usize = 8 + 8 + 32 + ProposalAction::LEN + 1 + 8 + 8 + 8 + 8 + 1;
}

#[account]
pub struct VoteReceipt {
    pub proposal: Pubkey,
    pub voter: Pubkey,
    pub support: bool,
    pub voting_power: u64,
}

impl VoteReceipt {
    pub const LEN: usize = 8 + 32 + 32 + 1 + 8;
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub struct AIControlledParameters {
    pub dynamic_burn_bps: u16,
    pub adaptive_entry_fee_bps: u16,
    pub redistribution_factor_bps: u16,
}

impl AIControlledParameters {
    pub const LEN: usize = 2 + 2 + 2;
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum ProposalAction {
    PauseProtocol,
    ResumeProtocol,
    UpdateAIParameters(AIControlledParameters),
}

impl ProposalAction {
    pub const LEN: usize = 1 + AIControlledParameters::LEN;
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq)]
pub enum ProposalStatus {
    Active,
    Succeeded,
    Failed,
    Executed,
}

#[event]
pub struct DepositEvent {
    pub user: Pubkey,
    pub gross_amount: u64,
    pub net_amount: u64,
    pub fee: u64,
    pub burn_amount: u64,
    pub reinvestment: u64,
}

#[event]
pub struct WithdrawEvent {
    pub user: Pubkey,
    pub amount: u64,
}

#[event]
pub struct ProposalCreatedEvent {
    pub proposal_id: u64,
    pub proposer: Pubkey,
}

#[event]
pub struct VoteCastEvent {
    pub proposal_id: u64,
    pub voter: Pubkey,
    pub support: bool,
    pub voting_power: u64,
}

#[event]
pub struct ProposalFinalizedEvent {
    pub proposal_id: u64,
    pub total_votes: u64,
    pub quorum_reached: bool,
    pub succeeded: bool,
}

#[event]
pub struct ProposalExecutedEvent {
    pub proposal_id: u64,
    pub executor: Pubkey,
}

fn validate_ai_params(params: &AIControlledParameters) -> Result<()> {
    require!(
        params.dynamic_burn_bps <= 500,
        ErrorCode::InvalidBurnRate
    );
    require!(
        params.adaptive_entry_fee_bps >= 50 && params.adaptive_entry_fee_bps <= 500,
        ErrorCode::InvalidFeeParameters
    );
    require!(
        params.redistribution_factor_bps >= 5_000 && params.redistribution_factor_bps <= 7_000,
        ErrorCode::InvalidRedistributionFactor
    );
    Ok(())
}

fn optimize_parameters(state: &mut State, tx_volume: u64, now: i64) -> Result<()> {
    let reserve_ratio_bps = if state.total_deposits == 0 {
        BPS_DENOMINATOR
    } else {
        state.security_reserve_amount
            .checked_mul(BPS_DENOMINATOR)
            .ok_or(ErrorCode::MathOverflow)?
            .checked_div(state.total_deposits)
            .ok_or(ErrorCode::MathOverflow)?
    };

    if reserve_ratio_bps < 2_500 {
        state.ai_params.adaptive_entry_fee_bps = (state.ai_params.adaptive_entry_fee_bps + 10).min(500);
    } else if state.ai_params.adaptive_entry_fee_bps > 50 {
        state.ai_params.adaptive_entry_fee_bps = state.ai_params.adaptive_entry_fee_bps.saturating_sub(5);
    }

    if tx_volume > 1_000_000 {
        state.ai_params.dynamic_burn_bps = (state.ai_params.dynamic_burn_bps + 10).min(500);
    } else if state.ai_params.dynamic_burn_bps > 50 {
        state.ai_params.dynamic_burn_bps = state.ai_params.dynamic_burn_bps.saturating_sub(5);
    }

    state.last_optimization_timestamp = now;
    Ok(())
}

fn calculate_voting_power(user: &UserPosition, now: i64) -> Result<u64> {
    let holding_bonus = if user.last_deposit > 0 && now > user.last_deposit {
        let held_seconds = (now - user.last_deposit) as u64;
        (held_seconds / (30 * 24 * 3600) as u64).min(12)
    } else {
        0
    };

    let loyalty_bonus = user.loyalty_points / 10;

    user.balance
        .checked_add(loyalty_bonus)
        .ok_or(ErrorCode::MathOverflow)?
        .checked_add(holding_bonus)
        .ok_or(ErrorCode::MathOverflow.into())
}

#[error_code]
pub enum ErrorCode {
    #[msg("Importo non valido.")]
    InvalidAmount,
    #[msg("Overflow aritmetico.")]
    MathOverflow,
    #[msg("Deposito troppo basso rispetto a fee, burn e reinvestimento.")]
    DepositTooLow,
    #[msg("Saldo insufficiente.")]
    InsufficientBalance,
    #[msg("Prelievo oltre il limite massimo.")]
    ExceedsMaxPayout,
    #[msg("Protocollo in pausa.")]
    ProtocolPaused,
    #[msg("Rate limit superato.")]
    RateLimitExceeded,
    #[msg("Rapporto di riserva insufficiente.")]
    InsecureReserveRatio,
    #[msg("Utente non autorizzato.")]
    Unauthorized,
    #[msg("Mint non valido.")]
    InvalidMint,
    #[msg("Soglia minima per creare la proposta non raggiunta.")]
    InsufficientProposalThreshold,
    #[msg("Proposta non attiva.")]
    ProposalNotActive,
    #[msg("Periodo di voto non valido.")]
    VotingPeriodInvalid,
    #[msg("Periodo di voto non ancora terminato.")]
    VotingPeriodNotEnded,
    #[msg("Proposta non approvata.")]
    ProposalNotSucceeded,
    #[msg("Proposta già eseguita.")]
    ProposalAlreadyExecuted,
    #[msg("Timelock non scaduto.")]
    TimelockNotExpired,
    #[msg("Burn rate non valido.")]
    InvalidBurnRate,
    #[msg("Fee parameters non validi.")]
    InvalidFeeParameters,
    #[msg("Redistribution factor non valido.")]
    InvalidRedistributionFactor,
    #[msg("Delega circolare non consentita.")]
    CircularDelegation,
}