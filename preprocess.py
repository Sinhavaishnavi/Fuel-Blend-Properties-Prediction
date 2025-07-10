import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FuelBlendEDA:
    def __init__(self, train_path='train.csv', test_path='test.csv'):
        """Initialize the EDA class with data paths"""
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.feature_cols = None
        self.target_cols = None
        self.blend_cols = None
        self.component_cols = None

    def load_data(self):
        """Load training and test datasets"""
        print("=" * 50)
        print("LOADING DATA")
        print("=" * 50)

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")

        # Define column groups based on dataset description
        self.blend_cols = [col for col in self.train_df.columns if 'fraction' in col.lower()]
        self.component_cols = [col for col in self.train_df.columns if 'Component' in col and 'Property' in col]
        self.target_cols = [col for col in self.train_df.columns if 'BlendProperty' in col]
        self.feature_cols = self.blend_cols + self.component_cols

        print(f"Blend composition columns: {len(self.blend_cols)}")
        print(f"Component property columns: {len(self.component_cols)}")
        print(f"Target columns: {len(self.target_cols)}")

        return self.train_df, self.test_df

    def basic_statistics(self):
        """Generate basic statistical summary"""
        print("\n" + "=" * 50)
        print("BASIC STATISTICS")
        print("=" * 50)

        # Training data statistics
        print("\nTRAINING DATA - FEATURE STATISTICS:")
        feature_stats = self.train_df[self.feature_cols].describe()
        print(feature_stats)

        print("\nTRAINING DATA - TARGET STATISTICS:")
        target_stats = self.train_df[self.target_cols].describe()
        print(target_stats)

        # Check for identical distributions between train and test features
        print("\nTRAIN vs TEST FEATURE DISTRIBUTION COMPARISON:")
        for col in self.feature_cols:
            if col in self.test_df.columns:
                train_mean = self.train_df[col].mean()
                test_mean = self.test_df[col].mean()
                train_std = self.train_df[col].std()
                test_std = self.test_df[col].std()

                mean_diff = abs(train_mean - test_mean) / train_mean * 100
                std_diff = abs(train_std - test_std) / train_std * 100

                if mean_diff > 5 or std_diff > 10:
                    print(f"{col}: Mean diff: {mean_diff:.2f}%, Std diff: {std_diff:.2f}%")

        return feature_stats, target_stats

    def check_missing_values(self):
        """Check for missing values and patterns"""
        print("\n" + "=" * 50)
        print("MISSING VALUES ANALYSIS")
        print("=" * 50)

        # Training data missing values
        train_missing = self.train_df.isnull().sum()
        train_missing_pct = (train_missing / len(self.train_df)) * 100

        print("TRAINING DATA MISSING VALUES:")
        missing_summary = pd.DataFrame({
            'Missing_Count': train_missing,
            'Missing_Percentage': train_missing_pct
        })
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
        print(missing_summary)

        # Test data missing values
        test_missing = self.test_df.isnull().sum()
        test_missing_pct = (test_missing / len(self.test_df)) * 100

        print("\nTEST DATA MISSING VALUES:")
        test_missing_summary = pd.DataFrame({
            'Missing_Count': test_missing,
            'Missing_Percentage': test_missing_pct
        })
        test_missing_summary = test_missing_summary[test_missing_summary['Missing_Count'] > 0]
        print(test_missing_summary)

        return missing_summary, test_missing_summary

    def detect_outliers(self):
        """Detect outliers using statistical methods"""
        print("\n" + "=" * 50)
        print("OUTLIER DETECTION")
        print("=" * 50)

        outlier_summary = {}

        # IQR method for outlier detection
        for col in self.feature_cols + self.target_cols:
            if col in self.train_df.columns:
                Q1 = self.train_df[col].quantile(0.25)
                Q3 = self.train_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.train_df[(self.train_df[col] < lower_bound) |
                                       (self.train_df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(self.train_df)) * 100

                outlier_summary[col] = {
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

        # Print outlier summary for columns with significant outliers
        print("OUTLIER SUMMARY (IQR Method):")
        for col, stats in outlier_summary.items():
            if stats['percentage'] > 5:  # Only show columns with >5% outliers
                print(f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")

        return outlier_summary

    def check_data_consistency(self):
        """Check data consistency and validation rules"""
        print("\n" + "=" * 50)
        print("DATA CONSISTENCY CHECKS")
        print("=" * 50)

        # Check if blend fractions sum to 1
        print("BLEND FRACTION VALIDATION:")
        blend_sums = self.train_df[self.blend_cols].sum(axis=1)
        print(f"Blend fractions sum statistics:")
        print(f"Mean: {blend_sums.mean():.6f}")
        print(f"Std: {blend_sums.std():.6f}")
        print(f"Min: {blend_sums.min():.6f}")
        print(f"Max: {blend_sums.max():.6f}")

        # Check for negative values in fractions
        negative_fractions = (self.train_df[self.blend_cols] < 0).sum().sum()
        print(f"Negative fraction values: {negative_fractions}")

        # Check for values > 1 in fractions
        over_one_fractions = (self.train_df[self.blend_cols] > 1).sum().sum()
        print(f"Fraction values > 1: {over_one_fractions}")

        # Check for zero variance features
        print("\nZERO VARIANCE FEATURES:")
        zero_var_features = []
        for col in self.feature_cols:
            if self.train_df[col].var() == 0:
                zero_var_features.append(col)
        print(f"Zero variance features: {zero_var_features}")

        return blend_sums, zero_var_features

    def correlation_analysis(self):
        """Analyze correlations between features and targets"""
        print("\n" + "=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        # Feature-target correlations
        print("FEATURE-TARGET CORRELATIONS (Top 10 for each target):")
        feature_target_corr = {}

        for target in self.target_cols:
            corr_with_target = self.train_df[self.feature_cols].corrwith(self.train_df[target])
            corr_with_target = corr_with_target.abs().sort_values(ascending=False)
            feature_target_corr[target] = corr_with_target.head(10)

            print(f"\n{target} - Top correlated features:")
            for feature, corr in corr_with_target.head(5).items():
                print(f"  {feature}: {corr:.4f}")

        # Inter-target correlations
        print("\nINTER-TARGET CORRELATIONS:")
        target_corr_matrix = self.train_df[self.target_cols].corr()

        # Find highly correlated target pairs
        high_corr_pairs = []
        for i in range(len(self.target_cols)):
            for j in range(i+1, len(self.target_cols)):
                corr = target_corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append((self.target_cols[i], self.target_cols[j], corr))

        print("Highly correlated target pairs (|r| > 0.7):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")

        return feature_target_corr, target_corr_matrix

    def feature_importance_analysis(self):
        """Analyze feature importance using mutual information"""
        print("\n" + "=" * 50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)

        feature_importance = {}

        for target in self.target_cols:
            # Calculate mutual information
            mi_scores = mutual_info_regression(
                self.train_df[self.feature_cols],
                self.train_df[target],
                random_state=42
            )

            mi_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)

            feature_importance[target] = mi_df

            print(f"\n{target} - Top 10 important features (Mutual Information):")
            for _, row in mi_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        return feature_importance

    def distribution_analysis(self):
        """Analyze feature and target distributions"""
        print("\n" + "=" * 50)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 50)

        # Normality tests for targets (important for XGBoost)
        print("NORMALITY TESTS FOR TARGETS (Shapiro-Wilk p-values):")
        normality_results = {}

        for target in self.target_cols:
            # Sample 5000 points for normality test (Shapiro-Wilk limitation)
            sample_size = min(5000, len(self.train_df))
            sample_data = self.train_df[target].sample(sample_size, random_state=42)

            stat, p_value = stats.shapiro(sample_data)
            normality_results[target] = p_value

            print(f"  {target}: p = {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Non-normal'})")

        # Skewness analysis
        print("\nSKEWNESS ANALYSIS:")
        skewness_results = {}

        for col in self.feature_cols + self.target_cols:
            if col in self.train_df.columns:
                skew = stats.skew(self.train_df[col])
                skewness_results[col] = skew

                if abs(skew) > 2:
                    print(f"  {col}: {skew:.4f} (Highly skewed)")

        return normality_results, skewness_results

    def preprocessing_recommendations(self):
        """Generate preprocessing recommendations for XGBoost"""
        print("\n" + "=" * 50)
        print("PREPROCESSING RECOMMENDATIONS FOR XGBOOST")
        print("=" * 50)

        recommendations = []

        # Check for missing values
        if self.train_df.isnull().sum().sum() > 0:
            recommendations.append("1. Handle missing values (XGBoost can handle them, but explicit handling may be better)")

        # Check for outliers
        outlier_summary = self.detect_outliers()
        high_outlier_cols = [col for col, stats in outlier_summary.items() if stats['percentage'] > 10]
        if high_outlier_cols:
            recommendations.append(f"2. Consider outlier treatment for: {high_outlier_cols[:5]}")

        # Check skewness
        _, skewness_results = self.distribution_analysis()
        highly_skewed = [col for col, skew in skewness_results.items() if abs(skew) > 2]
        if highly_skewed:
            recommendations.append(f"3. Consider log transformation for highly skewed features: {highly_skewed[:5]}")

        # Feature scaling (generally not needed for XGBoost, but can help)
        recommendations.append("4. Feature scaling not required for XGBoost, but can be beneficial for feature importance interpretation")

        # Feature selection
        recommendations.append("5. Consider feature selection based on mutual information scores")

        # Multicollinearity
        recommendations.append("6. XGBoost handles multicollinearity well, but removing highly correlated features may improve interpretability")

        print("RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  {rec}")

        return recommendations

    def create_processed_datasets(self):
        """Create processed datasets ready for XGBoost training"""
        print("\n" + "=" * 50)
        print("CREATING PROCESSED DATASETS")
        print("=" * 50)

        # Create copies for processing
        train_processed = self.train_df.copy()
        test_processed = self.test_df.copy()

        # Handle missing values (if any)
        if train_processed.isnull().sum().sum() > 0:
            print("Handling missing values...")
            train_processed = train_processed.fillna(train_processed.median())
            test_processed = test_processed.fillna(train_processed.median())

        # Ensure blend fractions sum to 1 (normalize if needed)
        blend_sums = train_processed[self.blend_cols].sum(axis=1)
        if not np.allclose(blend_sums, 1.0, atol=1e-6):
            print("Normalizing blend fractions...")
            train_processed[self.blend_cols] = train_processed[self.blend_cols].div(blend_sums, axis=0)

            test_blend_sums = test_processed[self.blend_cols].sum(axis=1)
            test_processed[self.blend_cols] = test_processed[self.blend_cols].div(test_blend_sums, axis=0)

        # Prepare final datasets
        X_train = train_processed[self.feature_cols]
        y_train = train_processed[self.target_cols]
        X_test = test_processed[self.feature_cols]

        print(f"Processed training features shape: {X_train.shape}")
        print(f"Processed training targets shape: {y_train.shape}")
        print(f"Processed test features shape: {X_test.shape}")

        return X_train, y_train, X_test

    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("STARTING COMPLETE EDA FOR FUEL BLEND DATASET")
        print("=" * 70)

        # Load data
        self.load_data()

        # Basic statistics
        self.basic_statistics()

        # Missing values
        self.check_missing_values()

        # Outlier detection
        self.detect_outliers()

        # Data consistency
        self.check_data_consistency()

        # Correlation analysis
        self.correlation_analysis()

        # Feature importance
        self.feature_importance_analysis()

        # Distribution analysis
        self.distribution_analysis()

        # Preprocessing recommendations
        self.preprocessing_recommendations()

        # Create processed datasets
        X_train, y_train, X_test = self.create_processed_datasets()

        print("\n" + "=" * 70)
        print("EDA COMPLETE - DATASETS READY FOR XGBOOST MODELING")
        print("=" * 70)

        return X_train, y_train, X_test

# Usage example
if __name__ == "__main__":
    # Initialize EDA class
    eda = FuelBlendEDA('/home/vipin/Programming/Hackathons/Fuel-Blend-Properties-Prediction/data/train.csv', '/home/vipin/Programming/Hackathons/Fuel-Blend-Properties-Prediction/data/test.csv')

    # Run complete EDA
    X_train, y_train, X_test = eda.run_complete_eda()

    # Save processed datasets
    X_train.to_csv('X_train_processed.csv', index=False)
    y_train.to_csv('y_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)

    print("\nProcessed datasets saved:")
    print("- X_train_processed.csv")
    print("- y_train_processed.csv")
    print("- X_test_processed.csv")
