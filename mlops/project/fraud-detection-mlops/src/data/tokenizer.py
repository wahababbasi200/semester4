"""
tokenizer.py
------------
Phase 3: Context-aware transaction tokenization with segment-specific
feature discretization.

Each receiver account's transaction history is converted into a token
sequence analogous to a NLP "sentence" that can be fed directly into
TF-IDF, Word2Vec, FastText, or DistilBERT embedding models.

Token sequence format:
  [CLS] [ACCT] <SEGMENT> <TXN_1_tokens> <TXN_2_tokens> ... [SEP]

Each transaction contributes 8 tokens (in order):
  [TXN] <TYPE> <AMT_*> <TIME_*> <DAY_*> <FREQ_*> <BAL_SENDER> <BAL_DEST>

Key EDA findings incorporated:
  - BAL_UNKNOWN token explicitly handles the PaySim destination balance
    data quality issue (both old/new dest balance = 0 for most fraud TRANSFERs)
  - BAL_EXACT_DEBIT is the dominant sender-side pattern in fraud (99.45%)
    and serves as a strong discriminative signal
  - Amount is right-skewed with fraud concentrating in high quantiles;
    quantile-based discretization captures this without assuming linearity
  - Temporal patterns differ between fraud and legitimate transactions;
    TIME_* and DAY_* tokens capture hour-of-day and day-of-week
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Vocabulary definition ────────────────────────────────────────────────────

SPECIAL_TOKENS  = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[ACCT]", "[TXN]"]
SEGMENT_TOKENS  = ["SEG_MICRO", "SEG_REGULAR", "SEG_HEAVY"]
ROLE_TOKENS     = ["SENDER", "RECEIVER"]
TYPE_TOKENS     = ["TRANSFER", "CASH_OUT"]
AMOUNT_TOKENS   = ["AMT_VERY_LOW", "AMT_LOW", "AMT_MEDIUM", "AMT_HIGH", "AMT_VERY_HIGH"]
TIME_TOKENS     = ["TIME_LATE_NIGHT", "TIME_MORNING", "TIME_AFTERNOON", "TIME_EVENING"]
DAY_TOKENS      = ["DAY_WEEKDAY", "DAY_WEEKEND"]
FREQ_TOKENS     = ["FREQ_LOW", "FREQ_MEDIUM", "FREQ_HIGH"]
BALANCE_TOKENS  = [
    "BAL_EXACT_DEBIT", "BAL_OVER_DEBIT", "BAL_UNDER_DEBIT",
    "BAL_NO_CHANGE", "BAL_UNKNOWN",
    "BAL_EXACT_CREDIT", "BAL_MISMATCH_CREDIT",
]

# Full ordered vocabulary (no duplicates)
_RAW_VOCAB = (
    SPECIAL_TOKENS + SEGMENT_TOKENS + ROLE_TOKENS + TYPE_TOKENS
    + AMOUNT_TOKENS + TIME_TOKENS + DAY_TOKENS + FREQ_TOKENS + BALANCE_TOKENS
)
VOCAB: List[str] = list(dict.fromkeys(_RAW_VOCAB))  # preserve order, remove dupes

_BAL_TOL = 1e-2  # tolerance for floating-point balance comparisons


# ─── Fitting ──────────────────────────────────────────────────────────────────

def fit_tokenizer(
    train_df: pd.DataFrame,
    amount_quantiles: Optional[List[float]] = None,
    frequency_quantiles: Optional[List[float]] = None,
    freq_lookback_steps: int = 24,
) -> Dict:
    """
    Compute amount and frequency quantile thresholds from the training set.

    All thresholds are derived ONLY from training transactions to prevent
    leakage; the same thresholds are applied unchanged to val/test data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training transactions (rows where split == 'train').
    amount_quantiles : list of 4 floats
        Boundaries for 5-bin amount discretization.
        Default: [0.20, 0.50, 0.80, 0.95]
    frequency_quantiles : list of 2 floats
        Boundaries for 3-bin frequency discretization.
        Default: [0.25, 0.75]
    freq_lookback_steps : int
        Lookback window (simulation steps) for counting recent transactions.

    Returns
    -------
    dict with keys:
      amount_thresholds, amount_quantiles,
      freq_thresholds, frequency_quantiles,
      freq_lookback_steps
    """
    if amount_quantiles is None:
        amount_quantiles = [0.20, 0.50, 0.80, 0.95]
    if frequency_quantiles is None:
        frequency_quantiles = [0.25, 0.75]

    assert len(amount_quantiles) == 4, "Need exactly 4 amount quantile boundaries"
    assert len(frequency_quantiles) == 2, "Need exactly 2 frequency quantile boundaries"

    # Amount thresholds from training amounts
    amount_thresholds = [
        float(np.quantile(train_df["amount"], q)) for q in amount_quantiles
    ]
    logger.info("Amount thresholds %s: %s", amount_quantiles, amount_thresholds)

    # Frequency thresholds: compute per-transaction recent counts on training set
    logger.info(
        "Computing recent frequency counts (lookback=%d steps) on %d training txns...",
        freq_lookback_steps, len(train_df),
    )
    recent_freqs = _compute_recent_frequencies(train_df, freq_lookback_steps)
    freq_thresholds = [
        float(np.quantile(recent_freqs, q)) for q in frequency_quantiles
    ]
    logger.info("Frequency thresholds %s: %s", frequency_quantiles, freq_thresholds)

    return {
        "amount_thresholds":  amount_thresholds,
        "amount_quantiles":   amount_quantiles,
        "freq_thresholds":    freq_thresholds,
        "frequency_quantiles": frequency_quantiles,
        "freq_lookback_steps": freq_lookback_steps,
    }


def _compute_recent_frequencies(df: pd.DataFrame, lookback: int) -> np.ndarray:
    """
    For each transaction, count prior transactions to the same nameDest
    within the lookback window [step - lookback, step - 1].

    Used only for fitting frequency quantile thresholds.
    """
    freqs = []
    for _, group in df.groupby("nameDest"):
        steps = group["step"].values
        for i, s in enumerate(steps):
            count = int(np.sum((steps[:i] >= s - lookback) & (steps[:i] < s)))
            freqs.append(count)
    return np.array(freqs)


# ─── Single-token mappers ─────────────────────────────────────────────────────

def _get_amount_token(amount: float, thresholds: Dict) -> str:
    """Map a transaction amount to an AMT_* token (5-bin quantile scheme)."""
    t = thresholds["amount_thresholds"]  # [q20, q50, q80, q95]
    if amount < t[0]:   return "AMT_VERY_LOW"
    elif amount < t[1]: return "AMT_LOW"
    elif amount < t[2]: return "AMT_MEDIUM"
    elif amount < t[3]: return "AMT_HIGH"
    else:               return "AMT_VERY_HIGH"


def _get_time_token(step: int) -> str:
    """Map simulation step to TIME_* token based on hour-of-day (step % 24)."""
    hour = step % 24
    if hour < 6:    return "TIME_LATE_NIGHT"
    elif hour < 12: return "TIME_MORNING"
    elif hour < 18: return "TIME_AFTERNOON"
    else:           return "TIME_EVENING"


def _get_day_token(step: int) -> str:
    """Map simulation step to DAY_* token based on day-of-week (step // 24 % 7)."""
    day = (step // 24) % 7
    return "DAY_WEEKDAY" if day < 5 else "DAY_WEEKEND"


def _get_freq_token(recent_count: int, thresholds: Dict) -> str:
    """Map a recent transaction count to a FREQ_* token (3-bin quantile scheme)."""
    t = thresholds["freq_thresholds"]  # [q25, q75]
    if recent_count <= t[0]:   return "FREQ_LOW"
    elif recent_count <= t[1]: return "FREQ_MEDIUM"
    else:                      return "FREQ_HIGH"


def _get_sender_balance_token(
    old_balance: float,
    new_balance: float,
    amount: float,
) -> str:
    """
    Classify the sender's balance change pattern relative to the transaction amount.

    EDA finding: 99.45% of fraud transactions show BAL_EXACT_DEBIT
    (oldbalanceOrg - amount ≈ newbalanceOrig), making this token highly
    discriminative for fraud detection.

    Token mapping:
      BAL_UNKNOWN     : both old and new balance = 0 (no information)
      BAL_EXACT_DEBIT : balance decrease ≈ transaction amount
      BAL_OVER_DEBIT  : balance decrease > transaction amount
      BAL_UNDER_DEBIT : 0 < balance decrease < transaction amount
      BAL_NO_CHANGE   : balance did not change
    """
    if old_balance == 0.0 and new_balance == 0.0:
        return "BAL_UNKNOWN"
    change = old_balance - new_balance  # positive = money left account
    tol = max(_BAL_TOL, 1e-6 * amount)
    if abs(change) < _BAL_TOL and amount > _BAL_TOL:
        return "BAL_NO_CHANGE"
    elif abs(change - amount) <= tol:
        return "BAL_EXACT_DEBIT"
    elif change > amount + tol:
        return "BAL_OVER_DEBIT"
    elif 0.0 < change < amount - tol:
        return "BAL_UNDER_DEBIT"
    else:
        return "BAL_NO_CHANGE"


def _get_dest_balance_token(
    old_balance: float,
    new_balance: float,
    amount: float,
) -> str:
    """
    Classify the destination's balance change pattern.

    EDA finding: Most fraud TRANSFER destinations have both old and new
    balance = 0 — a known PaySim simulation limitation. The explicit
    BAL_UNKNOWN token captures this artifact rather than masking it
    behind zero-valued numeric features.

    Token mapping:
      BAL_UNKNOWN       : both old and new balance = 0 (PaySim artifact)
      BAL_EXACT_CREDIT  : balance increase ≈ transaction amount
      BAL_MISMATCH_CREDIT: balance increase ≠ transaction amount
      BAL_NO_CHANGE     : balance did not change
    """
    if old_balance == 0.0 and new_balance == 0.0:
        return "BAL_UNKNOWN"
    change = new_balance - old_balance  # positive = money arrived
    tol = max(_BAL_TOL, 1e-6 * amount)
    if abs(change) < _BAL_TOL and amount > _BAL_TOL:
        return "BAL_NO_CHANGE"
    elif abs(change - amount) <= tol:
        return "BAL_EXACT_CREDIT"
    else:
        return "BAL_MISMATCH_CREDIT"


# ─── Transaction tokenization ─────────────────────────────────────────────────

def tokenize_transaction(
    row: pd.Series,
    thresholds: Dict,
    recent_count: int = 0,
) -> List[str]:
    """
    Convert a single transaction row into an ordered list of 8 tokens.

    Output: [[TXN], <TYPE>, <AMT>, <TIME>, <DAY>, <FREQ>, <BAL_SENDER>, <BAL_DEST>]

    Parameters
    ----------
    row : pd.Series
        One row from the filtered PaySim DataFrame.
    thresholds : dict
        Output of fit_tokenizer().
    recent_count : int
        Number of prior transactions to this nameDest in the lookback window.

    Returns
    -------
    list of 8 token strings.
    """
    return [
        "[TXN]",
        str(row["type"]),
        _get_amount_token(float(row["amount"]), thresholds),
        _get_time_token(int(row["step"])),
        _get_day_token(int(row["step"])),
        _get_freq_token(recent_count, thresholds),
        _get_sender_balance_token(
            float(row["oldbalanceOrg"]),
            float(row["newbalanceOrig"]),
            float(row["amount"]),
        ),
        _get_dest_balance_token(
            float(row["oldbalanceDest"]),
            float(row["newbalanceDest"]),
            float(row["amount"]),
        ),
    ]


# ─── Account sequence construction ───────────────────────────────────────────

def build_account_sequence(
    account_txns: pd.DataFrame,
    segment: str,
    thresholds: Dict,
    max_transactions: int = 30,
    freq_lookback_steps: int = 24,
) -> List[str]:
    """
    Build the full token sequence for one receiver account.

    The sequence encodes the account's transaction history as a "sentence"
    for downstream embedding models. Transactions are ordered chronologically
    (oldest first within the window); history is truncated to the most recent
    max_transactions to keep sequence length bounded.

    Format:
      [CLS] [ACCT] <SEGMENT> [TXN] <tok>... [TXN] <tok>... [SEP]

    Parameters
    ----------
    account_txns : pd.DataFrame
        All transactions for a single nameDest.
    segment : str
        Segment label from assign_segments() (SEG_MICRO / SEG_REGULAR / SEG_HEAVY).
    thresholds : dict
        Output of fit_tokenizer().
    max_transactions : int
        Maximum history length; keeps most recent transactions.
    freq_lookback_steps : int
        Window for frequency token computation.

    Returns
    -------
    list of token strings representing the full account sequence.
    """
    txns = account_txns.sort_values("step")
    if len(txns) > max_transactions:
        txns = txns.iloc[-max_transactions:]

    steps = txns["step"].values
    tokens: List[str] = ["[CLS]", "[ACCT]", segment]

    for i, (_, row) in enumerate(txns.iterrows()):
        current_step = int(row["step"])
        # Count prior transactions within lookback window
        recent_count = int(np.sum(
            (steps[:i] >= current_step - freq_lookback_steps) &
            (steps[:i] < current_step)
        ))
        tokens.extend(tokenize_transaction(row, thresholds, recent_count))

    tokens.append("[SEP]")
    return tokens


# ─── Dataset tokenization ─────────────────────────────────────────────────────

def tokenize_dataset(
    filtered_df: pd.DataFrame,
    account_segments: pd.Series,
    thresholds: Dict,
    account_labels: pd.Series,
    max_transactions: int = 30,
    freq_lookback_steps: int = 24,
) -> pd.DataFrame:
    """
    Tokenize all receiver accounts in a transaction DataFrame.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Filtered PaySim transactions (split-tagged; pass the appropriate subset).
    account_segments : pd.Series
        nameDest → segment string (output of assign_segments()).
    thresholds : dict
        Output of fit_tokenizer().
    account_labels : pd.Series
        nameDest → is_fraud_receiver (0 or 1).
    max_transactions : int
        Max history length per account.
    freq_lookback_steps : int
        Lookback window for frequency tokens.

    Returns
    -------
    pd.DataFrame indexed by account_id with columns:
      token_sequence (list), token_string (str), is_fraud_receiver, segment
    """
    records = []
    grouped = filtered_df.groupby("nameDest")
    total = len(grouped)

    for idx, (account_id, group) in enumerate(grouped):
        if idx % 50_000 == 0:
            logger.info("Tokenizing account %d / %d ...", idx, total)

        segment = account_segments.get(account_id, "SEG_MICRO")
        label   = int(account_labels.get(account_id, 0))
        tokens  = build_account_sequence(
            group, segment, thresholds,
            max_transactions=max_transactions,
            freq_lookback_steps=freq_lookback_steps,
        )
        records.append({
            "account_id":       account_id,
            "token_sequence":   tokens,
            "token_string":     " ".join(tokens),
            "is_fraud_receiver": label,
            "segment":          segment,
        })

    result = pd.DataFrame(records).set_index("account_id")
    n_fraud = int(result["is_fraud_receiver"].sum())
    logger.info(
        "Tokenized %d accounts (%d fraud, %.4f%%)",
        len(result), n_fraud, 100.0 * n_fraud / len(result),
    )
    return result


# ─── Config persistence ───────────────────────────────────────────────────────

def save_tokenizer_config(
    thresholds: Dict,
    vocab: List[str],
    path: str,
) -> None:
    """
    Persist tokenizer thresholds and vocabulary to JSON.

    This file is baked into the Docker image at build time so the serving
    container can reproduce the exact same tokenization without any
    training data or runtime downloads.

    Parameters
    ----------
    thresholds : dict
        Output of fit_tokenizer().
    vocab : list of str
        Full ordered vocabulary list (use the module-level VOCAB constant).
    path : str
        Output path for the JSON file.
    """
    config = {
        "thresholds":  thresholds,
        "vocabulary":  vocab,
        "vocab_size":  len(vocab),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Tokenizer config saved to %s (vocab_size=%d)", out_path, len(vocab))


def load_tokenizer_config(path: str) -> Dict:
    """Load a previously saved tokenizer config."""
    with open(path) as f:
        return json.load(f)
