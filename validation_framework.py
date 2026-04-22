"""
╔══════════════════════════════════════════════════════════════════════════╗
║  VAAYDO — Pipeline Validation Framework                                ║
║  17-Point System Integrity, Robustness & Deployment Readiness          ║
║  Hemrek Capital                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Sections:
  I    Data Integrity & Pipeline Validation     (§1–§2)
  II   Predictive Validation                    (§3–§4)
  III  Regime Detection Validation              (§5–§6)
  IV   Adaptive Weighting Validation            (§7–§8)
  V    Risk & Uncertainty Governance            (§9–§10)
  VI   Execution Realism                        (§11–§12)
  VII  Robustness & Anti-Overfit Tests          (§13–§15)
  VIII Monitoring & Governance                  (§17)
  IX   Capital Allocation Checklist             (Final Gate)
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import kstest, norm, kurtosis, skew
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
import warnings
import logging
import json
import copy

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    """Custom encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Single validation test result."""
    test_id: str
    test_name: str
    section: str
    passed: bool
    score: float  # 0.0 to 1.0
    severity: str  # CRITICAL, WARNING, INFO
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            'test_id': self.test_id, 'test_name': self.test_name,
            'section': self.section, 'passed': self.passed,
            'score': round(self.score, 4), 'severity': self.severity,
            'message': self.message, 'details': self.details,
            'timestamp': self.timestamp,
        }

@dataclass
class ValidationReport:
    """Complete validation report."""
    results: List[TestResult] = field(default_factory=list)
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    system_version: str = "4.0.0"

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_tests(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def critical_failures(self) -> List[TestResult]:
        return [r for r in self.results if not r.passed and r.severity == 'CRITICAL']

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0
        # Weighted: CRITICAL=3, WARNING=2, INFO=1
        weights = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
        total_w = sum(weights.get(r.severity, 1) for r in self.results)
        weighted_score = sum(r.score * weights.get(r.severity, 1) for r in self.results)
        return weighted_score / max(total_w, 1)

    @property
    def deployment_ready(self) -> bool:
        return len(self.critical_failures) == 0 and self.overall_score >= 0.70

    def summary(self) -> dict:
        return {
            'total': self.total_tests, 'passed': self.passed_tests,
            'failed': self.failed_tests, 'critical_failures': len(self.critical_failures),
            'overall_score': round(self.overall_score, 4),
            'deployment_ready': self.deployment_ready,
            'timestamp': self.run_timestamp,
        }

    def by_section(self) -> Dict[str, List[TestResult]]:
        sections = {}
        for r in self.results:
            sections.setdefault(r.section, []).append(r)
        return sections

    def to_json(self) -> str:
        return json.dumps({
            'summary': self.summary(),
            'results': [r.to_dict() for r in self.results],
        }, indent=2, cls=NpEncoder)


# ═══════════════════════════════════════════════════════════════════════════════
# I. DATA INTEGRITY & PIPELINE VALIDATION (§1–§2)
# ═══════════════════════════════════════════════════════════════════════════════

class DataIntegrityAudit:
    """§1: Data Integrity Audit — survivorship, lookahead, corporate actions, timestamps."""

    @staticmethod
    def test_survivorship_bias(df: pd.DataFrame, historical_symbols: List[str] = None) -> TestResult:
        """Check if only surviving securities are included (bias indicator)."""
        current_syms = set(df['Instrument'].unique())
        n_current = len(current_syms)

        # Heuristic: if we have historical symbol list, compare
        if historical_symbols:
            historical_set = set(historical_symbols)
            missing = historical_set - current_syms
            survival_rate = 1 - len(missing) / max(len(historical_set), 1)
            passed = survival_rate > 0.90  # Allow 10% attrition
            return TestResult(
                test_id="1.1", test_name="Survivorship Bias",
                section="I. Data Integrity", passed=passed,
                score=survival_rate, severity="CRITICAL",
                message=f"Survival rate: {survival_rate:.1%} ({len(missing)} symbols dropped)",
                details={'missing': list(missing)[:20], 'survival_rate': survival_rate}
            )

        # Without historical list, check data completeness as proxy
        completeness = df.notna().mean().mean()
        return TestResult(
            test_id="1.1", test_name="Survivorship Bias (Proxy)",
            section="I. Data Integrity", passed=completeness > 0.85,
            score=completeness, severity="WARNING",
            message=f"Data completeness: {completeness:.1%} (no historical symbol list provided)",
            details={'completeness': completeness, 'n_symbols': n_current}
        )

    @staticmethod
    def test_lookahead_bias(df: pd.DataFrame) -> TestResult:
        """Check for forward-looking data leakage in features."""
        issues = []

        # Test 1: Future prices shouldn't appear in current row's features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'future' in col.lower() or 'forward' in col.lower() or 'next' in col.lower():
                issues.append(f"Suspicious column name: {col}")

        # Test 2: Check if any feature has unrealistic correlation with price change
        if '% change' in df.columns and 'price' in df.columns:
            for col in numeric_cols:
                if col in ('% change', 'price', 'lot_size'):
                    continue
                try:
                    corr = df[col].corr(df['% change'])
                    if abs(corr) > 0.95:
                        issues.append(f"{col} has {corr:.3f} correlation with % change (possible leakage)")
                except Exception:
                    pass

        passed = len(issues) == 0
        return TestResult(
            test_id="1.2", test_name="Lookahead Bias",
            section="I. Data Integrity", passed=passed,
            score=1.0 if passed else 0.5,
            severity="CRITICAL",
            message=f"{'No leakage detected' if passed else f'{len(issues)} potential leakage(s)'}",
            details={'issues': issues}
        )

    @staticmethod
    def test_timestamp_alignment(df: pd.DataFrame) -> TestResult:
        """Validate timestamp consistency across all features."""
        # Check that all numeric features are from the same point in time
        # For cross-sectional data, check NaN patterns
        nan_pattern = df.isna().sum()
        inconsistent_cols = nan_pattern[nan_pattern > 0.5 * len(df)].index.tolist()
        n_complete_rows = (df.notna().all(axis=1)).sum()
        alignment_score = n_complete_rows / max(len(df), 1)

        return TestResult(
            test_id="1.4", test_name="Timestamp Alignment",
            section="I. Data Integrity", passed=alignment_score > 0.80,
            score=alignment_score, severity="WARNING",
            message=f"{n_complete_rows}/{len(df)} rows fully aligned ({alignment_score:.1%})",
            details={'incomplete_columns': inconsistent_cols, 'complete_rows': n_complete_rows}
        )

    @staticmethod
    def test_no_forward_fill_leakage(ohlcv_data: Dict[str, pd.DataFrame]) -> TestResult:
        """Check that forward-filling hasn't introduced leakage."""
        issues = []
        for sym, df in ohlcv_data.items():
            if df is None or df.empty:
                continue
            # Check for suspiciously long runs of identical values
            for col in ['Close', 'Volume']:
                if col not in df.columns:
                    continue
                runs = (df[col] != df[col].shift(1)).cumsum()
                max_run = runs.value_counts().max()
                if max_run > 10:  # 10+ identical values is suspicious
                    issues.append(f"{sym}.{col}: {max_run} consecutive identical values")

        passed = len(issues) <= len(ohlcv_data) * 0.05
        score = 1.0 - len(issues) / max(len(ohlcv_data), 1)
        return TestResult(
            test_id="1.6", test_name="Forward-Fill Leakage",
            section="I. Data Integrity", passed=passed,
            score=max(0, score), severity="CRITICAL",
            message=f"{len(issues)} forward-fill anomalies detected",
            details={'issues': issues[:20]}
        )


class FeatureValidation:
    """§2: Feature Construction Validation — stationarity, VIF, decay profiles."""

    @staticmethod
    def test_stationarity(df: pd.DataFrame, feature_cols: List[str] = None) -> TestResult:
        """ADF test for stationarity on key features."""
        from scipy.stats import kstest

        if feature_cols is None:
            raw_features = ['IVPercentile', 'rsi_daily', 'adx', '% change',
                           'kalman_trend', 'GARCH_Vol', 'RV_Composite']
            # Attempt to use stationary Z-scored features if available
            feature_cols = [f'z_{f}' if f'z_{f}' in df.columns else f for f in raw_features]

        stationary_count = 0
        non_stationary = []
        tested = 0

        for col in feature_cols:
            if col not in df.columns:
                continue
            values = df[col].dropna().values
            if len(values) < 20:
                continue
            tested += 1

            # Augmented Dickey-Fuller via simple regression test
            # H0: unit root (non-stationary)
            try:
                diff = np.diff(values)
                lag_values = values[:-1]
                if np.std(lag_values) < 1e-10:
                    continue
                # Simple regression: diff = alpha + beta * lag
                beta = np.corrcoef(diff, lag_values)[0, 1] * np.std(diff) / max(np.std(lag_values), 1e-10)
                # ADF critical values (approximate)
                t_stat = beta * np.sqrt(len(diff)) / max(np.std(diff - beta * lag_values), 1e-10)
                # 5% critical value for ADF ≈ -2.86
                if t_stat < -2.86:
                    stationary_count += 1
                else:
                    non_stationary.append(col)
            except Exception:
                continue

        score = stationary_count / max(tested, 1)
        return TestResult(
            test_id="2.1", test_name="Feature Stationarity (ADF)",
            section="I. Data Integrity", passed=score >= 0.60,
            score=score, severity="WARNING",
            message=f"{stationary_count}/{tested} features stationary at 5% level",
            details={'non_stationary': non_stationary, 'tested': tested}
        )

    @staticmethod
    def test_multicollinearity(df: pd.DataFrame, feature_cols: List[str] = None) -> TestResult:
        """Measure multicollinearity via VIF and condition number."""
        if feature_cols is None:
            feature_cols = ['IVPercentile', 'rsi_daily', 'adx', 'kalman_trend',
                           'GARCH_Vol', 'RV_Composite', 'PCR']

        available = [c for c in feature_cols if c in df.columns]
        if len(available) < 3:
            return TestResult(
                test_id="2.4", test_name="Multicollinearity (VIF)",
                section="I. Data Integrity", passed=True,
                score=1.0, severity="INFO",
                message="Too few features to test",
            )

        X = df[available].dropna()
        if len(X) < 10:
            return TestResult(
                test_id="2.4", test_name="Multicollinearity (VIF)",
                section="I. Data Integrity", passed=True,
                score=1.0, severity="INFO",
                message="Insufficient data for VIF test",
            )

        # Standardize
        X_std = (X - X.mean()) / X.std().replace(0, 1)

        # Condition number
        try:
            cond = np.linalg.cond(X_std.values)
        except Exception:
            cond = 999

        # VIF calculation
        vifs = {}
        for i, col in enumerate(available):
            try:
                others = [c for c in available if c != col]
                X_others = X_std[others].values
                y = X_std[col].values
                # R² from regression
                beta = np.linalg.lstsq(X_others, y, rcond=None)[0]
                y_hat = X_others @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_sq = 1 - ss_res / max(ss_tot, 1e-10)
                vif = 1 / max(1 - r_sq, 0.001)
                vifs[col] = round(vif, 2)
            except Exception:
                vifs[col] = 0

        high_vif = {k: v for k, v in vifs.items() if v > 10}
        passed = len(high_vif) == 0 and cond < 100
        score = 1.0 - len(high_vif) / max(len(vifs), 1)

        return TestResult(
            test_id="2.4", test_name="Multicollinearity (VIF)",
            section="I. Data Integrity", passed=passed,
            score=max(0, score), severity="WARNING",
            message=f"Condition number: {cond:.1f}, {len(high_vif)} features with VIF>10",
            details={'vifs': vifs, 'condition_number': round(cond, 2), 'high_vif': high_vif}
        )

    @staticmethod
    def test_feature_economic_rationale(df: pd.DataFrame) -> TestResult:
        """Verify each feature has sensible value ranges (economic rationale)."""
        checks = {
            'IVPercentile': (0, 100),
            'rsi_daily': (0, 100),
            'adx': (0, 100),
            'PCR': (0, 10),
            'GARCH_Vol': (0, 500),
            'RV_Composite': (0, 500),
            'kalman_trend': (-50, 50),
            '% change': (-50, 50),
        }

        violations = []
        for col, (lo, hi) in checks.items():
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            out_of_range = ((vals < lo) | (vals > hi)).sum()
            if out_of_range > 0:
                violations.append(f"{col}: {out_of_range} values outside [{lo}, {hi}]")

        passed = len(violations) == 0
        score = 1.0 - len(violations) / max(len(checks), 1)
        return TestResult(
            test_id="2.1b", test_name="Feature Economic Rationale",
            section="I. Data Integrity", passed=passed,
            score=max(0, score), severity="WARNING",
            message=f"{'All features in valid ranges' if passed else f'{len(violations)} range violations'}",
            details={'violations': violations}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# II. PREDICTIVE VALIDATION (§3–§4)
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveValidation:
    """§3–§4: Walk-forward testing and predictive power decomposition."""

    @staticmethod
    def test_walk_forward(score_fn: Callable, data_df: pd.DataFrame,
                          n_folds: int = 5, min_train: int = 50) -> TestResult:
        """Rolling walk-forward OOS validation.
        
        Args:
            score_fn: Function(train_df, test_df) -> dict with 'sharpe', 'returns'
            data_df: Full dataset sorted by time
            n_folds: Number of walk-forward folds
            min_train: Minimum training samples
        """
        n = len(data_df)
        fold_size = max(1, (n - min_train) // n_folds)

        oos_sharpes = []
        is_sharpes = []

        for fold in range(n_folds):
            train_end = min_train + fold * fold_size
            test_end = min(train_end + fold_size, n)
            if train_end >= n or test_end <= train_end:
                continue

            train = data_df.iloc[:train_end]
            test = data_df.iloc[train_end:test_end]

            try:
                result = score_fn(train, test)
                if result:
                    oos_sharpes.append(result.get('oos_sharpe', 0))
                    is_sharpes.append(result.get('is_sharpe', 0))
            except Exception:
                continue

        if not oos_sharpes:
            return TestResult(
                test_id="3.1", test_name="Walk-Forward OOS Sharpe",
                section="II. Predictive Validation", passed=False,
                score=0, severity="CRITICAL",
                message="Walk-forward test produced no results",
            )

        avg_oos = np.mean(oos_sharpes)
        avg_is = np.mean(is_sharpes) if is_sharpes else 1.0
        degradation = 1 - avg_oos / max(avg_is, 0.01) if avg_is > 0 else 1.0

        passed = avg_oos > 1.0 and degradation < 0.40
        score = min(1.0, max(0, avg_oos / 2.0))  # Normalize Sharpe to [0, 1]

        return TestResult(
            test_id="3.1", test_name="Walk-Forward OOS Sharpe",
            section="II. Predictive Validation", passed=passed,
            score=score, severity="CRITICAL",
            message=f"OOS Sharpe: {avg_oos:.2f}, IS→OOS degradation: {degradation:.1%}",
            details={
                'oos_sharpes': [round(s, 3) for s in oos_sharpes],
                'avg_oos_sharpe': round(avg_oos, 3),
                'avg_is_sharpe': round(avg_is, 3),
                'degradation': round(degradation, 3),
            }
        )

    @staticmethod
    def test_information_coefficient_stability(predictions: np.ndarray,
                                                actuals: np.ndarray,
                                                window: int = 20) -> TestResult:
        """Rolling IC stability test."""
        if len(predictions) < window * 2:
            return TestResult(
                test_id="3.3", test_name="IC Stability",
                section="II. Predictive Validation", passed=False,
                score=0, severity="WARNING",
                message="Insufficient data for IC stability test",
            )

        ics = []
        for i in range(window, len(predictions)):
            p = predictions[i - window:i]
            a = actuals[i - window:i]
            if np.std(p) > 1e-10 and np.std(a) > 1e-10:
                ic = np.corrcoef(p, a)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)

        if not ics:
            return TestResult(
                test_id="3.3", test_name="IC Stability",
                section="II. Predictive Validation", passed=False,
                score=0, severity="WARNING",
                message="Could not compute IC",
            )

        avg_ic = np.mean(ics)
        ic_std = np.std(ics)
        pct_positive = np.mean(np.array(ics) > 0)

        passed = avg_ic > 0.02 and pct_positive > 0.55
        score = min(1.0, max(0, avg_ic * 10))

        return TestResult(
            test_id="3.3", test_name="IC Stability",
            section="II. Predictive Validation", passed=passed,
            score=score, severity="WARNING",
            message=f"Avg IC: {avg_ic:.4f}, Std: {ic_std:.4f}, % Positive: {pct_positive:.1%}",
            details={'avg_ic': round(avg_ic, 4), 'ic_std': round(ic_std, 4),
                     'pct_positive': round(pct_positive, 3)}
        )

    @staticmethod
    def test_signal_marginal_contribution(df: pd.DataFrame,
                                           signal_cols: List[str],
                                           target_col: str = '% change') -> TestResult:
        """§4: Marginal contribution of each signal to predictive power."""
        if target_col not in df.columns:
            return TestResult(
                test_id="4.1", test_name="Signal Marginal Contribution",
                section="II. Predictive Validation", passed=True,
                score=0.5, severity="INFO",
                message="Target column not available for marginal test",
            )

        available = [c for c in signal_cols if c in df.columns]
        target = df[target_col].dropna()
        contributions = {}

        for col in available:
            vals = df.loc[target.index, col].dropna()
            common = vals.index.intersection(target.index)
            if len(common) < 20:
                continue
            # Information coefficient
            ic = np.corrcoef(vals.loc[common], target.loc[common])[0, 1]
            contributions[col] = round(abs(ic), 4) if not np.isnan(ic) else 0

        # Flag signals with near-zero contribution
        useless = [k for k, v in contributions.items() if v < 0.01]
        score = 1.0 - len(useless) / max(len(contributions), 1)

        return TestResult(
            test_id="4.1", test_name="Signal Marginal Contribution",
            section="II. Predictive Validation", passed=len(useless) <= len(contributions) * 0.3,
            score=max(0, score), severity="WARNING",
            message=f"{len(useless)}/{len(contributions)} signals with near-zero IC",
            details={'contributions': contributions, 'useless_signals': useless}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# III. REGIME DETECTION VALIDATION (§5–§6)
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeValidation:
    """§5–§6: Regime stability, transition realism, and usefulness tests."""

    @staticmethod
    def test_regime_stability(regime_states: list) -> TestResult:
        """§5: Average regime duration, transition frequency, entropy."""
        if len(regime_states) < 10:
            return TestResult(
                test_id="5.1", test_name="Regime Stability",
                section="III. Regime Validation", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient regime history for stability test",
            )

        # Extract dominant regime labels
        labels = []
        for r in regime_states:
            if hasattr(r, 'vol_probs') and hasattr(r, 'trend_probs'):
                # Dominant vol regime
                vol_label = int(np.argmax(r.vol_probs))
                trend_label = int(np.argmax(r.trend_probs))
                labels.append((vol_label, trend_label))
            else:
                labels.append((0, 0))

        # Compute run lengths
        changes = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
        avg_duration = len(labels) / max(changes, 1)
        flip_rate = changes / max(len(labels) - 1, 1)

        # Entropy of regime classification
        entropies = []
        for r in regime_states:
            if hasattr(r, 'entropy'):
                entropies.append(r.entropy)
        avg_entropy = np.mean(entropies) if entropies else 0.5

        # Too frequent (overfit) or never changes (decorative)
        # In universe scans, regimes vary by symbol, so lower threshold is acceptable.
        if avg_duration < 1.5:
            passed = False
            message = f"Regimes flip too frequently (avg {avg_duration:.1f} periods) — likely overfit"
        elif avg_duration > len(labels) * 0.95:
            passed = False
            message = f"Regimes almost never change (avg {avg_duration:.1f} periods) — likely decorative"
        else:
            passed = True
            message = f"Regime stability within bounds (avg {avg_duration:.1f} periods, flip rate: {flip_rate:.1%})"

        score = min(1.0, max(0, 1.0 - abs(flip_rate - 0.15) * 5))

        return TestResult(
            test_id="5.1", test_name="Regime Stability",
            section="III. Regime Validation", passed=passed,
            score=score, severity="CRITICAL",
            message=message,
            details={
                'avg_duration': round(avg_duration, 2),
                'flip_rate': round(flip_rate, 4),
                'avg_entropy': round(avg_entropy, 4),
                'total_regimes': len(labels),
                'total_changes': changes,
            }
        )

    @staticmethod
    def test_regime_usefulness(results_with_regime: List[dict],
                                results_without_regime: List[dict]) -> TestResult:
        """§6: Compare system performance with and without regime conditioning."""
        if not results_with_regime or not results_without_regime:
            return TestResult(
                test_id="6.1", test_name="Regime Usefulness",
                section="III. Regime Validation", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient data for regime usefulness comparison",
            )

        def _metrics(results):
            convictions = [r.get('conviction_score', 0) for r in results]
            pops = [r.get('pop', 0) for r in results]
            sharpes = [r.get('sharpe', 0) for r in results]
            return {
                'avg_conviction': np.mean(convictions),
                'avg_pop': np.mean(pops),
                'avg_sharpe': np.mean(sharpes),
                'n': len(results),
            }

        m_with = _metrics(results_with_regime)
        m_without = _metrics(results_without_regime)

        sharpe_diff = m_with['avg_sharpe'] - m_without['avg_sharpe']
        conviction_diff = m_with['avg_conviction'] - m_without['avg_conviction']

        # Statistical significance (basic t-test on convictions)
        cv_with = [r.get('conviction_score', 0) for r in results_with_regime]
        cv_without = [r.get('conviction_score', 0) for r in results_without_regime]
        if len(cv_with) >= 5 and len(cv_without) >= 5:
            t_stat, p_value = sp_stats.ttest_ind(cv_with, cv_without)
        else:
            t_stat, p_value = 0, 1.0

        significant = p_value < 0.05 and sharpe_diff > 0
        passed = significant or conviction_diff > 3  # At least 3 points improvement

        score = min(1.0, max(0, sharpe_diff + 0.5))

        return TestResult(
            test_id="6.1", test_name="Regime Usefulness",
            section="III. Regime Validation", passed=passed,
            score=score, severity="CRITICAL",
            message=f"Sharpe Δ: {sharpe_diff:+.3f}, Conviction Δ: {conviction_diff:+.1f}, p={p_value:.4f}",
            details={
                'with_regime': m_with, 'without_regime': m_without,
                'sharpe_diff': round(sharpe_diff, 4),
                'p_value': round(p_value, 4),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# IV. ADAPTIVE WEIGHTING VALIDATION (§7–§8)
# ═══════════════════════════════════════════════════════════════════════════════

class WeightingValidation:
    """§7–§8: Weight drift, model competition, redundancy."""

    @staticmethod
    def test_weight_stability(weight_history: List[np.ndarray]) -> TestResult:
        """§7: Track weight variance, detect oscillation vs freezing."""
        if len(weight_history) < 5:
            return TestResult(
                test_id="7.1", test_name="Weight Drift Stability",
                section="IV. Adaptive Weighting", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient weight history",
            )

        weights = np.array(weight_history)
        # Weight variance over time
        weight_var = np.var(weights, axis=0)
        avg_var = np.mean(weight_var)

        # Detect oscillation: high variance in diffs
        diffs = np.diff(weights, axis=0)
        oscillation = np.mean(np.std(diffs, axis=0))

        # Detect freezing: very low variance
        frozen = np.sum(weight_var < 1e-6)

        # Half-life of weight changes
        autocorrs = []
        for j in range(weights.shape[1]):
            w = weights[:, j]
            if len(w) > 2 and np.std(w) > 1e-10:
                ac = np.corrcoef(w[:-1], w[1:])[0, 1]
                if not np.isnan(ac) and ac > 0:
                    hl = -np.log(2) / np.log(max(ac, 0.01))
                    autocorrs.append(hl)

        avg_half_life = np.mean(autocorrs) if autocorrs else 0

        issues = []
        if oscillation > 0.1:
            issues.append("Oscillating weights detected")
        if frozen > weights.shape[1] * 0.5:
            issues.append(f"{frozen}/{weights.shape[1]} weights frozen")
        if avg_half_life < 2:
            issues.append("Weights update too aggressively")

        passed = len(issues) == 0
        score = max(0, 1.0 - len(issues) * 0.33)

        return TestResult(
            test_id="7.1", test_name="Weight Drift Stability",
            section="IV. Adaptive Weighting", passed=passed,
            score=score, severity="WARNING",
            message=f"Avg variance: {avg_var:.4f}, oscillation: {oscillation:.4f}, half-life: {avg_half_life:.1f}",
            details={'avg_variance': round(avg_var, 4), 'oscillation': round(oscillation, 4),
                     'avg_half_life': round(avg_half_life, 2), 'issues': issues}
        )

    @staticmethod
    def test_model_redundancy(model_outputs: Dict[str, np.ndarray]) -> TestResult:
        """§8: Check if one model dominates or models are redundant."""
        if len(model_outputs) < 2:
            return TestResult(
                test_id="8.1", test_name="Model Redundancy",
                section="IV. Adaptive Weighting", passed=True,
                score=0.5, severity="INFO",
                message="Less than 2 models to compare",
            )

        names = list(model_outputs.keys())
        n_models = len(names)

        # Cross-correlation matrix
        corr_matrix = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(i + 1, n_models):
                a, b = model_outputs[names[i]], model_outputs[names[j]]
                min_len = min(len(a), len(b))
                if min_len > 5:
                    corr = np.corrcoef(a[:min_len], b[:min_len])[0, 1]
                    corr_matrix[i, j] = corr_matrix[j, i] = corr if not np.isnan(corr) else 0

        # Average off-diagonal correlation
        mask = ~np.eye(n_models, dtype=bool)
        avg_corr = np.mean(np.abs(corr_matrix[mask]))

        # Dominance: one model's predictions cover >80% of ensemble output
        redundant_pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if abs(corr_matrix[i, j]) > 0.90:
                    redundant_pairs.append((names[i], names[j], round(corr_matrix[i, j], 3)))

        passed = avg_corr < 0.70 and len(redundant_pairs) == 0
        score = max(0, 1.0 - avg_corr)

        return TestResult(
            test_id="8.1", test_name="Model Redundancy",
            section="IV. Adaptive Weighting", passed=passed,
            score=score, severity="WARNING",
            message=f"Avg model correlation: {avg_corr:.3f}, {len(redundant_pairs)} redundant pairs",
            details={'avg_correlation': round(avg_corr, 3), 'redundant_pairs': redundant_pairs}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# V. RISK & UNCERTAINTY GOVERNANCE (§9–§10)
# ═══════════════════════════════════════════════════════════════════════════════

class RiskGovernance:
    """§9–§10: Full distribution testing and stress testing."""

    @staticmethod
    def test_return_distribution(returns: np.ndarray) -> TestResult:
        """§9: Evaluate skew, kurtosis, tail ratio, CDaR."""
        if len(returns) < 30:
            return TestResult(
                test_id="9.1", test_name="Return Distribution",
                section="V. Risk Governance", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient return data",
            )

        sk = float(skew(returns))
        kt = float(kurtosis(returns))
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        # Tail ratio: |95th percentile| / |5th percentile|
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = abs(p95) / max(abs(p5), 1e-10)

        # Worst rolling 6-month (126 trading days)
        window = min(126, len(returns) // 2)
        if window > 5:
            rolling_cumulative = pd.Series(returns).rolling(window).sum()
            worst_6m = float(rolling_cumulative.min()) if not rolling_cumulative.isna().all() else 0
        else:
            worst_6m = float(np.sum(returns))

        # CDaR: Conditional Drawdown at Risk (expected drawdown in worst 5% of cases)
        cum_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = running_max - cum_returns
        cdar_5 = float(np.percentile(drawdowns, 95))  # 5% worst drawdowns

        issues = []
        if kt > 6:
            issues.append(f"Excess kurtosis ({kt:.1f}) — fat tails")
        if sk < -1:
            issues.append(f"Negative skew ({sk:.2f}) — left tail risk")
        if tail_ratio < 0.8:
            issues.append(f"Unfavorable tail ratio ({tail_ratio:.2f})")

        passed = len(issues) == 0
        score = max(0, 1.0 - len(issues) * 0.25)

        return TestResult(
            test_id="9.1", test_name="Return Distribution",
            section="V. Risk Governance", passed=passed,
            score=score, severity="WARNING",
            message=f"Skew: {sk:.2f}, Kurtosis: {kt:.1f}, Tail ratio: {tail_ratio:.2f}, CDaR(5%): {cdar_5:.4f}",
            details={
                'skew': round(sk, 3), 'kurtosis': round(kt, 3),
                'tail_ratio': round(tail_ratio, 3), 'cdar_5pct': round(cdar_5, 4),
                'worst_6m': round(worst_6m, 4), 'issues': issues,
            }
        )

    @staticmethod
    def test_stress_scenarios(score_fn: Callable, base_data: dict,
                               scenarios: Dict[str, dict] = None) -> TestResult:
        """§10: Stress test under historical crisis scenarios."""
        if scenarios is None:
            scenarios = {
                '2008 Liquidity Collapse': {'vol_mult': 3.0, 'price_shock': -0.40, 'spread_mult': 5.0},
                '2020 Vol Spike': {'vol_mult': 4.0, 'price_shock': -0.30, 'spread_mult': 3.0},
                'Vol Compression': {'vol_mult': 0.3, 'price_shock': 0.0, 'spread_mult': 1.0},
                'Inflation Shock': {'vol_mult': 1.8, 'price_shock': -0.15, 'spread_mult': 2.0},
                'Correlation Breakdown': {'vol_mult': 2.0, 'price_shock': -0.20, 'spread_mult': 2.5},
            }

        results = {}
        survived = 0

        for scenario_name, params in scenarios.items():
            stressed = copy.deepcopy(base_data)
            # Apply stress
            if 'ATMIV' in stressed:
                stressed['ATMIV'] *= params.get('vol_mult', 1.0)
            if 'price' in stressed:
                stressed['price'] *= (1 + params.get('price_shock', 0))
            if 'RV_Composite' in stressed:
                stressed['RV_Composite'] *= params.get('vol_mult', 1.0)
            if 'IVPercentile' in stressed:
                stressed['IVPercentile'] = min(99, stressed['IVPercentile'] * params.get('vol_mult', 1.0))

            try:
                result = score_fn(stressed)
                # Check if system reduced exposure
                conv = result.get('conviction_score', 50) if isinstance(result, dict) else 50
                results[scenario_name] = {
                    'conviction': conv,
                    'reduced_exposure': conv < 40,
                    'survived': conv > 0,
                }
                if conv < 60:  # System correctly reduced conviction
                    survived += 1
            except Exception as e:
                results[scenario_name] = {'error': str(e), 'survived': False}

        survival_rate = survived / max(len(scenarios), 1)
        passed = survival_rate >= 0.80  # Must handle 4/5 scenarios

        return TestResult(
            test_id="10.1", test_name="Stress Test Survivability",
            section="V. Risk Governance", passed=passed,
            score=survival_rate, severity="CRITICAL",
            message=f"Survived {survived}/{len(scenarios)} stress scenarios",
            details={'scenarios': results, 'survival_rate': round(survival_rate, 3)}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VI. EXECUTION REALISM (§11–§12)
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionRealism:
    """§11–§12: Transaction cost modeling and latency sensitivity."""

    @staticmethod
    def test_transaction_costs(strategies: List[dict], cost_model: str = 'pessimistic') -> TestResult:
        """§11: Impact of realistic transaction costs on strategy performance."""
        if not strategies:
            return TestResult(
                test_id="11.1", test_name="Transaction Cost Impact",
                section="VI. Execution Realism", passed=True,
                score=0.5, severity="INFO",
                message="No strategies to test",
            )

        # Pessimistic cost assumptions for Indian options
        cost_per_lot = {
            'brokerage': 40,       # Flat ₹20 per order × 2 legs minimum
            'stt': 0.0005,        # STT on sell side
            'exchange_txn': 0.00053,
            'gst': 0.18,          # GST on brokerage
            'sebi_fee': 0.000001,
            'stamp_duty': 0.00003,
            'slippage_bps': 5,    # 5 bps slippage
        }

        if cost_model == 'pessimistic':
            cost_per_lot['slippage_bps'] = 10  # Higher slippage

        viable_after_cost = 0
        total = len(strategies)
        cost_impacts = []

        for s in strategies:
            premium = abs(s.get('net_credit', 0)) * s.get('lot_size', 1)
            spot = s.get('price', 100)
            lot = s.get('lot_size', 1)
            n_legs = len(s.get('_result', type('', (), {'legs': [1, 2]})()).legs) if hasattr(s.get('_result'), 'legs') else 2

            # Total cost per trade
            brokerage = cost_per_lot['brokerage'] * n_legs
            stt = premium * cost_per_lot['stt']
            txn = premium * cost_per_lot['exchange_txn']
            gst = brokerage * cost_per_lot['gst']
            slippage = spot * lot * cost_per_lot['slippage_bps'] / 10000 * n_legs
            total_cost = brokerage + stt + txn + gst + slippage

            # Impact on profitability
            max_profit = s.get('mp_lot', 0)
            if max_profit > 0:
                cost_pct = total_cost / max_profit * 100
                cost_impacts.append(cost_pct)
                if max_profit - total_cost > 0:
                    viable_after_cost += 1

        viability_rate = viable_after_cost / max(total, 1)
        avg_cost_impact = np.mean(cost_impacts) if cost_impacts else 0

        passed = viability_rate >= 0.70 and avg_cost_impact < 30
        score = min(1.0, viability_rate)

        return TestResult(
            test_id="11.1", test_name="Transaction Cost Impact",
            section="VI. Execution Realism", passed=passed,
            score=score, severity="CRITICAL",
            message=f"{viable_after_cost}/{total} strategies viable after costs (avg impact: {avg_cost_impact:.1f}%)",
            details={'viability_rate': round(viability_rate, 3),
                     'avg_cost_impact_pct': round(avg_cost_impact, 2)}
        )

    @staticmethod
    def test_latency_sensitivity(score_fn: Callable, base_data: dict,
                                  delay_bars: List[int] = [1, 2, 3]) -> TestResult:
        """§12: Performance with delayed execution."""
        base_result = score_fn(base_data)
        base_conv = base_result.get('conviction_score', 50) if isinstance(base_result, dict) else 50

        degradations = {}
        for delay in delay_bars:
            # Simulate delayed entry by slightly worsening conditions
            delayed = copy.deepcopy(base_data)
            # Price drift against us
            if 'price' in delayed:
                delayed['price'] *= (1 + 0.002 * delay)  # 0.2% per bar delay
            if 'ATMIV' in delayed:
                delayed['ATMIV'] *= (1 - 0.01 * delay)   # IV decay

            try:
                result = score_fn(delayed)
                conv = result.get('conviction_score', 50) if isinstance(result, dict) else 50
                degradation = (base_conv - conv) / max(base_conv, 1)
                degradations[f"{delay}_bar"] = round(degradation, 4)
            except Exception:
                degradations[f"{delay}_bar"] = 1.0

        avg_degradation = np.mean(list(degradations.values()))
        collapsed = any(v > 0.50 for v in degradations.values())

        passed = not collapsed and avg_degradation < 0.30
        score = max(0, 1.0 - avg_degradation)

        return TestResult(
            test_id="12.1", test_name="Latency Sensitivity",
            section="VI. Execution Realism", passed=passed,
            score=score, severity="WARNING",
            message=f"Avg degradation: {avg_degradation:.1%}, collapsed: {collapsed}",
            details={'degradations': degradations, 'base_conviction': round(base_conv, 2)}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VII. ROBUSTNESS & ANTI-OVERFIT (§13–§15)
# ═══════════════════════════════════════════════════════════════════════════════

class RobustnessTests:
    """§13–§15: Parameter perturbation, Monte Carlo resampling, simplicity benchmark."""

    @staticmethod
    def test_parameter_perturbation(score_fn: Callable, base_data: dict,
                                     n_perturbations: int = 50) -> TestResult:
        """§13: Randomly perturb parameters and check sensitivity."""
        base_result = score_fn(base_data)
        base_conv = base_result.get('conviction_score', 50) if isinstance(base_result, dict) else 50

        perturbed_scores = []
        for _ in range(n_perturbations):
            perturbed = copy.deepcopy(base_data)
            # Perturb each numeric field by ±10%
            for key in ['ATMIV', 'IVPercentile', 'RV_Composite', 'GARCH_Vol',
                        'rsi_daily', 'adx', 'kalman_trend']:
                if key in perturbed and isinstance(perturbed[key], (int, float)):
                    perturbed[key] *= np.random.uniform(0.90, 1.10)

            try:
                result = score_fn(perturbed)
                conv = result.get('conviction_score', 50) if isinstance(result, dict) else 50
                perturbed_scores.append(conv)
            except Exception:
                continue

        if not perturbed_scores:
            return TestResult(
                test_id="13.1", test_name="Parameter Perturbation",
                section="VII. Robustness", passed=False,
                score=0, severity="CRITICAL",
                message="All perturbations failed",
            )

        mean_perturbed = np.mean(perturbed_scores)
        std_perturbed = np.std(perturbed_scores)
        cv = std_perturbed / max(mean_perturbed, 1)  # Coefficient of variation

        passed = cv < 0.25  # Less than 25% variation
        score = max(0, 1.0 - cv)

        return TestResult(
            test_id="13.1", test_name="Parameter Perturbation",
            section="VII. Robustness", passed=passed,
            score=score, severity="CRITICAL",
            message=f"CV: {cv:.3f}, Base: {base_conv:.1f}, Mean perturbed: {mean_perturbed:.1f} ± {std_perturbed:.1f}",
            details={'base_conviction': round(base_conv, 2),
                     'mean_perturbed': round(mean_perturbed, 2),
                     'std_perturbed': round(std_perturbed, 2),
                     'coefficient_of_variation': round(cv, 4)}
        )

    @staticmethod
    def test_monte_carlo_resampling(returns: np.ndarray, n_paths: int = 1000) -> TestResult:
        """§14: Bootstrap resampling for ruin probability, Sharpe distribution."""
        if len(returns) < 30:
            return TestResult(
                test_id="14.1", test_name="MC Path Resampling",
                section="VII. Robustness", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient return data for MC test",
            )

        sharpes = []
        max_dds = []
        final_values = []

        for _ in range(n_paths):
            # Bootstrap resample with replacement
            idx = np.random.choice(len(returns), size=len(returns), replace=True)
            path = returns[idx]

            # Sharpe
            if np.std(path) > 1e-10:
                sharpes.append(np.mean(path) / np.std(path) * np.sqrt(252))

            # Max drawdown
            cum = np.cumsum(path)
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dds.append(float(np.max(dd)))

            # Final value
            final_values.append(float(np.sum(path)))

        p_ruin = np.mean(np.array(final_values) < -0.20)  # >20% total loss
        median_sharpe = np.median(sharpes) if sharpes else 0
        median_dd = np.median(max_dds) if max_dds else 0
        p_positive_sharpe = np.mean(np.array(sharpes) > 0) if sharpes else 0

        passed = p_ruin < 0.05 and p_positive_sharpe > 0.60
        score = max(0, (1.0 - p_ruin) * p_positive_sharpe)

        return TestResult(
            test_id="14.1", test_name="MC Path Resampling",
            section="VII. Robustness", passed=passed,
            score=score, severity="CRITICAL",
            message=f"P(ruin): {p_ruin:.2%}, Median Sharpe: {median_sharpe:.2f}, P(Sharpe>0): {p_positive_sharpe:.1%}",
            details={
                'p_ruin': round(p_ruin, 4),
                'median_sharpe': round(median_sharpe, 3),
                'sharpe_5th': round(np.percentile(sharpes, 5), 3) if sharpes else 0,
                'sharpe_95th': round(np.percentile(sharpes, 95), 3) if sharpes else 0,
                'median_max_dd': round(median_dd, 4),
                'p_positive_sharpe': round(p_positive_sharpe, 3),
            }
        )

    @staticmethod
    def test_simplicity_benchmark(system_sharpe: float, system_dd: float,
                                   benchmark_sharpes: Dict[str, float] = None) -> TestResult:
        """§15: Compare against simple baselines."""
        if benchmark_sharpes is None:
            benchmark_sharpes = {
                'Equal Weight': 0.40,
                'Risk Parity': 0.55,
                'Momentum Only': 0.35,
                'Buy & Hold Nifty': 0.50,
            }

        beats = {}
        for name, bm_sharpe in benchmark_sharpes.items():
            beats[name] = system_sharpe > bm_sharpe

        n_beaten = sum(beats.values())
        total = len(beats)
        passed = n_beaten >= total * 0.6  # Beat at least 60% of benchmarks

        score = n_beaten / max(total, 1)
        best_bm = max(benchmark_sharpes.values())

        return TestResult(
            test_id="15.1", test_name="Simplicity Benchmark",
            section="VII. Robustness", passed=passed,
            score=score, severity="CRITICAL",
            message=f"System Sharpe ({system_sharpe:.2f}) beats {n_beaten}/{total} baselines (best baseline: {best_bm:.2f})",
            details={'system_sharpe': round(system_sharpe, 3), 'benchmarks': benchmark_sharpes,
                     'beats': beats}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIII. PERFORMANCE MONITORING (§17)
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """§17: Real-time drift detection and kill-switch logic."""

    def __init__(self):
        self.ic_history: List[float] = []
        self.entropy_history: List[float] = []
        self.regime_confidence_history: List[float] = []
        self.allocation_history: List[np.ndarray] = []
        self._kill_switch_triggered = False
        self._throttle_level = 0.0  # 0 = full, 1 = fully throttled

    def update(self, ic: float, entropy: float, regime_confidence: float,
               allocations: np.ndarray = None):
        """Record new observation."""
        self.ic_history.append(ic)
        self.entropy_history.append(entropy)
        self.regime_confidence_history.append(regime_confidence)
        if allocations is not None:
            self.allocation_history.append(allocations)

    def check_ic_decay(self, window: int = 20, threshold: float = -0.02) -> TestResult:
        """Monitor information coefficient for decay."""
        if len(self.ic_history) < window * 2:
            return TestResult(
                test_id="17.1", test_name="IC Decay Monitor",
                section="VIII. Monitoring", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient IC history",
            )

        recent = np.mean(self.ic_history[-window:])
        older = np.mean(self.ic_history[-2 * window:-window])
        decay = recent - older

        passed = decay > threshold
        score = max(0, min(1, (decay - threshold) / 0.1 + 0.5))

        return TestResult(
            test_id="17.1", test_name="IC Decay Monitor",
            section="VIII. Monitoring", passed=passed,
            score=score, severity="CRITICAL" if not passed else "INFO",
            message=f"IC trend: {decay:+.4f} (recent: {recent:.4f}, older: {older:.4f})",
            details={'recent_ic': round(recent, 4), 'older_ic': round(older, 4), 'decay': round(decay, 4)}
        )

    def check_entropy_trend(self, window: int = 10) -> TestResult:
        """Monitor system entropy for deterioration."""
        if len(self.entropy_history) < window * 2:
            return TestResult(
                test_id="17.2", test_name="Entropy Trend",
                section="VIII. Monitoring", passed=True,
                score=0.5, severity="INFO",
                message="Insufficient entropy history",
            )

        recent = np.mean(self.entropy_history[-window:])
        older = np.mean(self.entropy_history[-2 * window:-window])
        trend = recent - older

        passed = recent < 0.70 and trend < 0.15
        score = max(0, 1.0 - recent)

        return TestResult(
            test_id="17.2", test_name="Entropy Trend",
            section="VIII. Monitoring", passed=passed,
            score=score, severity="WARNING" if not passed else "INFO",
            message=f"Entropy: {recent:.3f} (trend: {trend:+.3f})",
            details={'current_entropy': round(recent, 4), 'trend': round(trend, 4)}
        )

    def check_kill_switch(self) -> dict:
        """Evaluate kill-switch conditions."""
        conditions = {
            'ic_collapse': False,
            'entropy_critical': False,
            'confidence_collapse': False,
            'allocation_unstable': False,
        }

        if len(self.ic_history) >= 10:
            recent_ic = np.mean(self.ic_history[-10:])
            conditions['ic_collapse'] = recent_ic < -0.05

        if len(self.entropy_history) >= 5:
            recent_entropy = np.mean(self.entropy_history[-5:])
            conditions['entropy_critical'] = recent_entropy > 0.85

        if len(self.regime_confidence_history) >= 5:
            recent_conf = np.mean(self.regime_confidence_history[-5:])
            conditions['confidence_collapse'] = recent_conf < 0.20

        if len(self.allocation_history) >= 5:
            recent_allocs = np.array(self.allocation_history[-5:])
            alloc_var = np.mean(np.var(recent_allocs, axis=0))
            conditions['allocation_unstable'] = alloc_var > 0.10

        n_triggered = sum(conditions.values())
        self._kill_switch_triggered = n_triggered >= 2
        self._throttle_level = n_triggered / max(len(conditions), 1)

        return {
            'kill_switch': self._kill_switch_triggered,
            'throttle_level': round(self._throttle_level, 2),
            'conditions': conditions,
            'recommendation': 'HALT' if self._kill_switch_triggered else (
                'REDUCE' if self._throttle_level > 0.25 else 'NORMAL'
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# IX. CAPITAL ALLOCATION CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════

class CapitalAllocationGate:
    """Final deployment readiness checklist before capital allocation."""

    CHECKLIST = [
        ('1yr_oos_walkforward', '1 Year OOS Walk-Forward', 'CRITICAL'),
        ('live_shadow', 'Live Shadow Validation', 'CRITICAL'),
        ('stress_survived', 'Stress Test Survivability', 'CRITICAL'),
        ('cost_robust', 'Cost Robustness', 'CRITICAL'),
        ('model_redundancy', 'Model Redundancy Verified', 'WARNING'),
        ('regime_value', 'Regime Value Statistically Confirmed', 'WARNING'),
        ('psychological', 'Psychological Survivability Assessed', 'INFO'),
    ]

    @staticmethod
    def evaluate(validation_report: ValidationReport,
                 live_shadow_days: int = 0,
                 psychological_assessed: bool = False) -> TestResult:
        """Evaluate capital allocation readiness."""
        checks = {}

        # Map validation results to checklist items
        for r in validation_report.results:
            if 'Walk-Forward' in r.test_name:
                checks['1yr_oos_walkforward'] = r.passed
            elif 'Stress' in r.test_name:
                checks['stress_survived'] = r.passed
            elif 'Transaction Cost' in r.test_name:
                checks['cost_robust'] = r.passed
            elif 'Redundancy' in r.test_name:
                checks['model_redundancy'] = r.passed
            elif 'Regime Usefulness' in r.test_name:
                checks['regime_value'] = r.passed

        # External checks
        checks['live_shadow'] = live_shadow_days >= 30
        checks['psychological'] = psychological_assessed

        passed_critical = all(
            checks.get(item[0], False)
            for item in CapitalAllocationGate.CHECKLIST
            if item[2] == 'CRITICAL'
        )
        total_passed = sum(checks.values())
        total = len(CapitalAllocationGate.CHECKLIST)

        score = total_passed / max(total, 1)
        passed = passed_critical and score >= 0.80

        return TestResult(
            test_id="GATE", test_name="Capital Allocation Gate",
            section="IX. Capital Allocation", passed=passed,
            score=score, severity="CRITICAL",
            message=f"{'✅ APPROVED' if passed else '❌ NOT READY'} — {total_passed}/{total} checks passed",
            details={'checks': checks, 'critical_passed': passed_critical,
                     'live_shadow_days': live_shadow_days}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationRunner:
    """Orchestrates all validation tests and produces comprehensive report."""

    def __init__(self):
        self.report = ValidationReport()
        self.monitor = PerformanceMonitor()

    def run_data_integrity(self, df: pd.DataFrame,
                           ohlcv_data: Dict[str, pd.DataFrame] = None,
                           historical_symbols: List[str] = None):
        """Run all data integrity tests."""
        # §1: Data Integrity
        self.report.results.append(
            DataIntegrityAudit.test_survivorship_bias(df, historical_symbols))
        self.report.results.append(
            DataIntegrityAudit.test_lookahead_bias(df))
        self.report.results.append(
            DataIntegrityAudit.test_timestamp_alignment(df))
        if ohlcv_data:
            self.report.results.append(
                DataIntegrityAudit.test_no_forward_fill_leakage(ohlcv_data))

        # §2: Feature Validation
        self.report.results.append(
            FeatureValidation.test_stationarity(df))
        self.report.results.append(
            FeatureValidation.test_multicollinearity(df))
        self.report.results.append(
            FeatureValidation.test_feature_economic_rationale(df))

    def run_predictive_validation(self, df: pd.DataFrame,
                                   score_fn: Callable = None,
                                   predictions: np.ndarray = None,
                                   actuals: np.ndarray = None):
        """Run predictive validation tests."""
        if score_fn:
            self.report.results.append(
                PredictiveValidation.test_walk_forward(score_fn, df))

        if predictions is not None and actuals is not None:
            self.report.results.append(
                PredictiveValidation.test_information_coefficient_stability(predictions, actuals))

        # Use z_ prefixed stationary signals if available, else raw
        signal_cols = ['IVPercentile', 'rsi_daily', 'adx', 'kalman_trend',
                       'GARCH_Vol', 'RV_Composite', 'PCR', 'GARCH_Persistence']
        stationary_signals = [f'z_{c}' for c in signal_cols if f'z_{c}' in df.columns]
        target_signals = stationary_signals if stationary_signals else [c for c in signal_cols if c in df.columns]
        
        self.report.results.append(
            PredictiveValidation.test_signal_marginal_contribution(df, target_signals))

    def run_regime_validation(self, regime_states: list,
                               results_with: List[dict] = None,
                               results_without: List[dict] = None):
        """Run regime detection validation tests."""
        self.report.results.append(
            RegimeValidation.test_regime_stability(regime_states))
        if results_with and results_without:
            self.report.results.append(
                RegimeValidation.test_regime_usefulness(results_with, results_without))

    def run_risk_governance(self, returns: np.ndarray = None,
                            score_fn: Callable = None,
                            base_data: dict = None):
        """Run risk and uncertainty governance tests."""
        if returns is not None:
            self.report.results.append(
                RiskGovernance.test_return_distribution(returns))
        if score_fn and base_data:
            self.report.results.append(
                RiskGovernance.test_stress_scenarios(score_fn, base_data))

    def run_execution_realism(self, strategies: List[dict] = None,
                               score_fn: Callable = None,
                               base_data: dict = None):
        """Run execution realism tests."""
        if strategies:
            self.report.results.append(
                ExecutionRealism.test_transaction_costs(strategies))
        if score_fn and base_data:
            self.report.results.append(
                ExecutionRealism.test_latency_sensitivity(score_fn, base_data))

    def run_robustness(self, score_fn: Callable = None,
                       base_data: dict = None,
                       returns: np.ndarray = None,
                       system_sharpe: float = None):
        """Run robustness and anti-overfit tests."""
        if score_fn and base_data:
            self.report.results.append(
                RobustnessTests.test_parameter_perturbation(score_fn, base_data))
        if returns is not None:
            self.report.results.append(
                RobustnessTests.test_monte_carlo_resampling(returns))
        if system_sharpe is not None:
            self.report.results.append(
                RobustnessTests.test_simplicity_benchmark(system_sharpe, 0))

    def run_monitoring(self, regime_states: list):
        """Run real-time drift detection and monitoring tests."""
        monitor = PerformanceMonitor()
        # Populate history from current state
        for rs in regime_states:
            # Assuming rs is a FuzzyRegime object or has entropy attr
            ent = getattr(rs, 'entropy', 0.5)
            # IC and confidence aren't tracked historically here, using defaults
            monitor.update(ic=0.05, entropy=ent, regime_confidence=0.5)
        
        self.report.results.append(monitor.check_entropy_trend())

    def run_capital_gate(self, live_shadow_days: int = 0,
                         psychological_assessed: bool = False):
        """Run final capital allocation gate."""
        self.report.results.append(
            CapitalAllocationGate.evaluate(self.report, live_shadow_days, psychological_assessed))

    def run_all(self, df: pd.DataFrame, strategies: List[dict] = None,
                regime_states: list = None, ohlcv_data: Dict[str, pd.DataFrame] = None,
                returns: np.ndarray = None, score_fn: Callable = None,
                base_data: dict = None) -> ValidationReport:
        """Run all available validation tests."""
        self.report = ValidationReport()

        # I. Data Integrity
        self.run_data_integrity(df, ohlcv_data)

        # II. Predictive Validation
        self.run_predictive_validation(df)

        # III. Regime Validation
        if regime_states:
            self.run_regime_validation(regime_states)

        # V. Risk Governance
        if returns is not None:
            self.run_risk_governance(returns, score_fn, base_data)

        # VI. Execution Realism
        if strategies:
            self.run_execution_realism(strategies, score_fn, base_data)

        # VII. Robustness
        if score_fn and base_data:
            self.run_robustness(score_fn, base_data, returns)

        # IX. Capital Gate (always last)
        self.run_capital_gate()

        return self.report
