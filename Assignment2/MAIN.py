#### WORK FLOW ####
# Run these in order:
# 1. dataCleaner.py: clean and filter raw data. Outputs to relevant data.
# 2. teamGenerator.py: creates random teams from players based on stats.
# 3. ANN.py: defines neural network class for training and prediction.
# 4. MAIN.py: main script to run the full pipeline. Finds the most optimal team. from a lot of different combinations.

import os
import sys
import pandas as pd
from itertools import combinations
import streamlit as st
from tensorflow import keras
sys.path.append(os.path.dirname(__file__))
import ANN

script_dir = os.path.dirname(os.path.abspath(__file__))

### teamGenerator.py ###
class BasketballTeamEvaluator:
    """
    Evaluates basketball teams based on basketball knowledge and principles.
    Scores teams on balance, complementary skills, and overall strength.
    """
    
    def __init__(self):
        # The weights determine how important certain features are. I played around with these to get different
        # scores and these seems to be effective values.
        self.weights = {
            'balance': 0.15, # General and physical stats: player_height, player_weigt, gp
            'strength': 0.15, # Personal stats: ts_pct, pts, reb, net_rating, oreb, dreb
            'complementary': 0.70 # Team interaction stats: ast, usg_pct, ast_pct
        }
        {# 1. player_name — Name of the player 
        # 2. team_abbreviation — Abbreviated name of the team the player played for (at the end of the season) -
        # 3. age — Age of the player -
        # 4. player_height — Height of the player (in centimeters)-
        # 5. player_weight — Weight of the player (in kilograms)-
        # 6. gp — Games played throughout the season-
        # 7. pts — Average number of points scored-
        # 8. reb — Average number of rebounds grabbed-
        # 9. ast — Average number of assists distributed
        # 10. net_rating — Team's point differential per 100 possessions while the player is on the court-
        # 11. oreb_pct — Percentage of available offensive rebounds the player grabbed while he was on the floor
        # 12. dreb_pct — Percentage of available defensive rebounds the player grabbed while he was on the floor
        # 13. usg_pct — Percentage of team plays used by the player while he was on the floor ((FGA + Possession Ending FTA + TO) / POSS)
        # 14. ts_pct — Measure of the player's shooting efficiency that takes into account free throws, 2 and 3 point shots (PTS / (2*(FGA + 0.44 * FTA)))
        # 15. ast_pct — Percentage of teammate field goals the player assisted while he was on the floor
        # 16. season — NBA season -
        }
        
    
    def evaluate_team(self, team_df: pd.DataFrame) -> Dict:
        """
        Evaluate a 5-player team and return detailed scoring.
        
        Args:
            team_df: DataFrame with 5 players and their stats
            
        Returns:
            Dictionary with scores and team features
        """
        
        # Calculate component scores
        balance_score = self._calculate_balance_score(team_df)
        strength_score = self._calculate_strength_score(team_df)
        complementary_score = self._calculate_complementary_score(team_df)
        
        # Overall weighted score
        total_score = (
            balance_score * self.weights['balance'] +
            strength_score * self.weights['strength'] +
            complementary_score * self.weights['complementary']
        )
        
        # Generate team features for neural network
        features = self._generate_team_features(team_df)
        
        return {
            'total_score': total_score,
            'balance_score': balance_score,
            'strength_score': strength_score,
            'complementary_score': complementary_score,
            'features': features,
            'players': team_df['player_name'].tolist()
        }
    
    def _calculate_balance_score(self, team_df: pd.DataFrame) -> float:
        """Calculate how balanced the team is across different roles."""
        score = 0.0
        
        ### Height balance
        heights = team_df['player_height'].values
        height_std = np.std(heights)
        height_balance = min(height_std / 15.0, 1.0)  # Normalize by typical std
        score += height_balance * 0.4 # importance of the height feature

        ### Weight balance - variety in physicality
        weights = team_df['player_weight'].values
        weight_std = np.std(weights)
        weight_balance = min(weight_std / 20.0, 1.0)
        score += weight_balance * 0.4

        ### Games played - prefer players who played significant minutes
        games_played = team_df['gp']
        gp_score = np.mean(games_played) / 82.0   # linear normalization
        score += gp_score * 0.2
        
        return score
    
    def _calculate_strength_score(self, team_df: pd.DataFrame) -> float:
        """Calculate overall team strength in key areas, including rebounding and games played."""
        score = 0.0

        # ------------------------------
        # Offensive efficiency (True Shooting %)
        avg_ts = team_df['ts_pct'].fillna(0.5).mean()
        ts_score = min(max((avg_ts - 0.45) / 0.15, 0), 1)  # Scale 0.45-0.6 → 0-1
        score += ts_score * 0.1  # slightly lower to leave room for other stats

        # Scoring punch
        avg_pts = team_df['pts'].mean()
        pts_score = min(avg_pts / 20.0, 1.0)  # normalize to ~20 PPG
        score += pts_score * 0.3

         # Total rebounds
        avg_reb = team_df['reb'].fillna(0).mean()
        reb_score = min(avg_reb / 10.0, 1.0)  # normalize, assume 10 rpg is excellent
        score += reb_score * 0.15

        # Rebounding efficiency
        # Offensive rebounds
        oreb = team_df['oreb_pct'].fillna(0).mean()  # average oreb%
        oreb_score = min(oreb / 0.3, 1.0)  # assume 30% is elite
        score += oreb_score * 0.15

        # Defensive rebounds
        dreb = team_df['dreb_pct'].fillna(0).mean()  # average dreb%
        dreb_score = min(dreb / 0.8, 1.0)  # assume 80% is elite
        score += dreb_score * 0.15

        # Overall impact (net rating)
        avg_net = team_df['net_rating'].fillna(0).mean()
        net_score = min(max((avg_net + 10) / 20, 0), 1)  # Scale -10 → +10 to 0-1
        score += net_score * 0.15

        return score

    def _calculate_complementary_score(self, team_df: pd.DataFrame) -> float:
        """Calculate how well players complement each other based on usage and playmaking."""

        score = 0.0

        # ------------------------------
        # Usage rate balance: avoid 5 ball-dominant players
        usage_rates = team_df['usg_pct'].fillna(0.15).values
        avg_usage = np.mean(usage_rates)
        usage_balance = 1.0 - min(abs(avg_usage - 0.22) / 0.1, 1.0)  # Optimal ~22%
        score += usage_balance * 0.4  # weight for usage balance

        # ------------------------------
        # Playmaking distribution: ensure some players handle passing
        playmakers_pct = sum(team_df['ast_pct'].fillna(0.1) > 0.2)  # high assist %
        playmaker_balance_pct = min(playmakers_pct / 2.0, 1.0)      # want ~1-2 playmakers
        score += playmaker_balance_pct * 0.3  # weight for ast_pct

        # ------------------------------
        # Total assists: reward teams with more passing overall
        total_ast = team_df['ast'].fillna(0).sum()
        # Normalize: assume 20-40 assists per game is good for a 5-player team
        ast_score = min(total_ast / 40.0, 1.0)
        score += ast_score * 0.3  # weight for raw assists

        return score

    
    def _generate_team_features(self, team_df: pd.DataFrame) -> List[float]:
        """Generate numerical features for neural network input."""
        features = []
        
        # Basic averaged stats
        features.extend([
            team_df['pts'].mean(),
            team_df['reb'].mean(), 
            team_df['ast'].mean(),
            team_df['ts_pct'].fillna(0.5).mean(),
            team_df['net_rating'].fillna(0).mean(),
            team_df['usg_pct'].fillna(0.15).mean(),
            team_df['ast_pct'].fillna(0.1).mean(),
        ])
        
        # Variability measures (balance)
        features.extend([
            team_df['pts'].std(),
            team_df['reb'].std(),
            team_df['player_height'].std(),
            team_df['age'].std(),
            team_df['usg_pct'].fillna(0.15).std(),
        ])
        
        # Team composition features
        features.extend([
            team_df['player_height'].max() - team_df['player_height'].min(),  # Height range
            team_df['age'].max() - team_df['age'].min(),  # Age range
            sum(team_df['pts'] > 15),  # High scorers
            sum(team_df['ast'] > 4),   # Playmakers  
            sum(team_df['reb'] > 7),   # Strong rebounders
            sum(team_df['ts_pct'].fillna(0.4) > 0.55),  # Good shooters
        ])
        
        # Physical attributes
        features.extend([
            team_df['player_height'].mean(),
            team_df['player_weight'].mean(),
            team_df['age'].mean(),
        ])
        
        return features

def generate_teams_and_scores(csv_file="relevant_data.csv", sample_size=5000, random_seed=42):
    """
    Generate random team combinations and score them.
    
    Args:
        csv_file: Path to cleaned player data
        sample_size: Number of random teams to generate and score
        random_seed: For reproducible results
    """
    # Load player data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} players from {csv_file}")
    
    # Initialize evaluator
    evaluator = BasketballTeamEvaluator()
    
    # Generate random team combinations
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    team_data = []
    feature_data = []
    scores = []
    
    print(f"Generating and scoring {sample_size} random teams...")
    
    for i in range(sample_size):
        # Randomly select 5 players
        team_indices = random.sample(range(len(df)), 5)
        team_df = df.iloc[team_indices].copy()
        
        # Evaluate team
        evaluation = evaluator.evaluate_team(team_df)
        
        # Store results
        team_data.append({
            'team_id': i,
            'players': evaluation['players'],
            'total_score': evaluation['total_score'],
            'balance_score': evaluation['balance_score'],
            'strength_score': evaluation['strength_score'],
            'complementary_score': evaluation['complementary_score']
        })
        
        feature_data.append(evaluation['features'])
        scores.append(evaluation['total_score'])
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{sample_size} teams")
    
    # Convert to DataFrames
    teams_df = pd.DataFrame(team_data)
    
    # Create features DataFrame
    feature_names = [
        'avg_pts', 'avg_reb', 'avg_ast', 'avg_ts_pct', 'avg_net_rating', 'avg_usg_pct', 'avg_ast_pct',
        'std_pts', 'std_reb', 'std_height', 'std_age', 'std_usg_pct', 
        'height_range', 'age_range', 'high_scorers', 'playmakers', 'rebounders', 'good_shooters',
        'avg_height', 'avg_weight', 'avg_age'
    ]
    
    features_df = pd.DataFrame(feature_data, columns=feature_names)
    features_df['team_score'] = scores
    
    # Save results
    teams_df.to_csv(os.path.join(script_dir,'team_evaluations.csv'), index=False)
    features_df.to_csv(os.path.join(script_dir,'team_features.csv'), index=False)
    
    print(f"\nResults saved:")
    print(f"  - team_evaluations.csv: Team compositions and scores")
    print(f"  - team_features.csv: Features for neural network training")
    print(f"\nScore distribution:")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std:  {np.std(scores):.3f}")
    print(f"  Min:  {np.min(scores):.3f}")
    print(f"  Max:  {np.max(scores):.3f}")
    
    # Show best team
    best_idx = np.argmax(scores)
    print(f"\nBest team (Score: {scores[best_idx]:.3f}):")
    for player in team_data[best_idx]['players']:
        print(f"  - {player}")
    
    return teams_df, features_df


### FUNCTION FOR FINDING OPTIMAL TEAM ###
def find_optimal_team(csv_file="relevant_data.csv", model=None, optional_max_teams=500):
    """
    Find the optimal team using the trained neural network.
    
    Args:
        csv_file: Path to player data
        model: Trained BasketballNeuralNetwork instance
    
    Returns:
        Best team DataFrame and score
    """
    # Load player data
    df = pd.read_csv(os.path.join(script_dir, csv_file))
    print(f"Finding optimal team from {len(df)} players...")
    
    evaluator = BasketballTeamEvaluator() # used to evaluate teams.
    
    best_score = 0
    best_team = None
    best_features = None
    
    # Generate all possible 5-player combinations
    # This will generate over 75,000,000 combinations for 100 players. There is no way our computers can handle that,
    # so fall backs have been implemented, such as only using 500 teams.
    total_combinations = len(list(combinations(range(len(df)), 5)))
    st.write(f"Total possible teams: {total_combinations:,}")
    
    if total_combinations > 100000:
        st.write(f"Too many combinations! Sampling {optional_max_teams} random teams...")
        # Sample random combinations
        import random
        random.seed(42)
        team_combinations = [random.sample(range(len(df)), 5) for _ in range(optional_max_teams)]
    else:
        print("Evaluating all possible combinations...")
        team_combinations = list(combinations(range(len(df)), 5))
    
    st.write(f"Evaluating {len(team_combinations):,} teams...")
    
    for i, team_indices in enumerate(team_combinations):
        # Get team
        team_df = df.iloc[list(team_indices)]
        
        # Generate features
        features = evaluator._generate_team_features(team_df)
        
        # Predict score
        predicted_score = model.predict_team_score(features)
        
        # Track best team
        if predicted_score > best_score:
            best_score = predicted_score
            best_team = team_df.copy()
            best_features = features
        
        # Progress update
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(team_combinations):,} teams. Best score so far: {best_score:.4f}")
    
    print(f"\nOPTIMAL TEAM FOUND!")
    print(f"Predicted Score: {best_score:.4f}/1")
    print(f"\nPlayers:")
    for idx, player in best_team.iterrows():
        print(f"  • {player['player_name']}")
    
    # Save optimal team (not needed but nice to have)
    best_team.to_csv(os.path.join(script_dir, 'optimal_team.csv'), index=False)
    # print(f"\nOptimal team saved to 'optimal_team.csv'")
    
    return best_team, best_score

def main():
    MODEL_PATH = os.path.join(script_dir, "basketball_team_model.keras")

    st.write("SEARCHING FOR OPTIMAL TEAM...")

    # Always initialize your wrapper class
    nn = ANN.NeuralNetwork(input_dim=21)

    # If the .keras model file exists, load it into the wrapper
    if os.path.exists(MODEL_PATH):
        st.write("Loading existing model from .keras file...")
        nn.load_model()  # This loads into your wrapper class
    else:
        # Build and train new model
        nn.build_model(hidden_layers=[64, 32, 16], dropout_rate=0.3, learning_rate=0.001)
        
        X_train, X_test, y_train, y_test = nn.prepare_data(features_file="team_features.csv", test_size=0.2)
        
        nn.train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=64, patience=10)
        
        nn.plot_training_history()
        nn.save_model()

    # Now nn is always your wrapper class with predict_team_score()
    optimal_team, optimal_score = find_optimal_team(
        csv_file="relevant_data.csv",
        model=nn  # This now has predict_team_score()
    )

    return nn, optimal_team, optimal_score


if __name__ == "__main__":
    # Run the complete pipeline
    st.title("🏀 Optimal Basketball Team Finder")

    st.markdown("""
                # GitHub Repository
                [Link to GitHub Repository](https://github.com/nolanrd04/CST-435-Assignment2-ANNv1.1)
                I wasn't sure if you wanted the whole thing submitted in a zip file, or just the streamlit link.
                # AI usage
                Claude AI helped a lot in generating code to calculate team scores and features. It did not include everything,
                So I spent a while going through and adjusting weights, fixing bugs, and adding comments. I do not know a lot
                about basketball, so having claude help seemed reasonable. Additionally, The output seems to actually make sense.
                Both ChatGPT and Claude were used in helping to create the pipeline and clean things up.

                # Project explanation

                ### This project uses basketball player data to build the most optimal team of 5 players using a neural network.
                ### Here are the steps involved:
                1. **Data Cleaning**: The raw player data is cleaned and filtered to keep relevant statistics. Only 100 players are picked.
                2. **Team Feature Generation**: Random teams are generated and their features are calculated. Picked from the 100 players.
                3. **Rules established**: There are rules made in order to determine what makes a good team.
                There are three categories that each have their own weights, and they are comprised of individual
                features that also have their own weights. The weights of the featres were determined by my general
                knowledge of basketball and trial and error. The weights of the categories were determined by trial
                and error.
                4. **Neural Network Training**: A neural network is built and trained to predict team quality scores based on the generated features.
                The neural network learns from the diffeerent combiations of teams and their scores.
                5. **Optimal Team Search**: The trained model is used to evaluate many possible 5-player combinations to find the best team, given the 100 players.
                """)
    
    model, best_team, score = main()
    
    st.subheader("Best Team Found")
    st.write(f"Predicted Score: {score:.4f}/1")
    st.dataframe(best_team)