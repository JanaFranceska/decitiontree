from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from dtreeviz import dtreeviz
from dtreeviz.models.shadow_decision_tree import ShadowDecTree  # Needed for new dtreeviz API
from dtreeviz import model

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("final.csv")

# Fixed features
#fixed_features = ['Gender','DivAv_AttraktedAv', 'IndentifyAvatar', 'Competition', 'Cooperation', 'SMS']
categorical_features = ['Gender','AttrAvatar', 'IndentifyAvatar', 'Sex', 'BodyType', 'SocialContext']

numerical_features = ['Competition', 'Cooperation', 'SMS']
#target = 'TargetVariable'

df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

encoded_categorical_columns = [col for col in df_encoded.columns if any(cat in col for cat in categorical_features)]


# Save the decision tree visualization as PDF
def save_decision_tree_as_pdf(graph, target_variable):
    
    # Generate the PDF file name using the target variable
    file_name = f"pdf/{target_variable}_decision_tree"
    
    # Render the decision tree as a PDF with the file name based on the target variable
    graph.render(file_name)  # Saves as target_variable_decision_tree.pdf

    print(f"Decision tree saved as {file_name}.pdf")

def save_tree_as_svg(graph, output_file):

    graph.render(output_file, format='svg')


def save_decision_tree_as_png(tree_model, preprocessor, feature_names, target_variable):
    file_name = f"png/{target_variable}_decision_tree"
    graph.render(file_name, format="png")  # Saves as target_variable_decision_tree.png

    print(f"Decision tree saved as {file_name}.png")
    
    return f"{file_name}.png"  # Return the file path to use in the HTML

# HTML form to submit inputs
@app.route('/')
def form():
    return render_template('index.html')


# Endpoint to handle form submission and run the model
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    Gender = request.form['Gender']
    AttrAvatar = request.form['AttrAvatar']
    IndentifyAvatar = request.form['IndentifyAvatar']
    Competition = float(request.form['Competition'])
    Cooperation = float(request.form['Cooperation'])
    SMS = float(request.form['SMS'])
    target_variable = request.form['target_variable']

    # Create fixed_values dictionary from form input
    fixed_values_example = {
        'Gender': Gender,
        'AttrAvatar': AttrAvatar,
        'IndentifyAvatar': IndentifyAvatar,
        'Competition': Competition,
        'Cooperation': Cooperation,
        'SMS': SMS
    }
    
    
    # Define the feature matrix X and target vector y
    X = df_encoded[encoded_categorical_columns + numerical_features]
    y = df_encoded[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the decision tree classifier
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Get the feature names after one-hot encoding
    full_feature_names = list(X.columns)

    # Generate the DOT data for Graphviz
    dot_data = export_graphviz(decision_tree, out_file=None,
                           feature_names=full_feature_names,
                           filled=True, rounded=True, special_characters=True, label='all')
                        
    #dot_data = export_graphviz(decision_tree, out_file=None,
    #                       feature_names=full_feature_names,class_names=decision_tree.classes_.astype(str),
     #                      filled=True, rounded=True, special_characters=True, label='all')

    # Create a Graphviz source object and display the tree
    graph = graphviz.Source(dot_data)

    
    # Evaluate the model
    y_pred = decision_tree.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Print evaluation metrics
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    
    # Find the best combination using the existing model
    best_combination = find_best_combination(fixed_values_example, full_feature_names, decision_tree, df_encoded)
    
    
    # create pdf
    save_decision_tree_as_pdf(graph,target_variable)
    #save_decision_tree_as_pdf(decision_tree, preprocessor, all_features, target_variable)

    # create png
    #tree_image_path = save_decision_tree_as_png(graph, target_variable)


    # creat svg
    save_tree_as_svg(graph, 'static/decision_tree')

    #tree_html_path=tree_html_path,
    return render_template('result.html', best_combination=best_combination.to_dict(), tree_svg_path='static/decision_tree.svg',target_variable=target_variable,fixed_values_example=fixed_values_example)
    
    
# Function to find the best combination for optimizable features
def find_best_combination(fixed_values, feature_names, model, df_encoded):
    # Generate all possible combinations of the optimizable features
    sex_values = df['Sex'].unique()
    bodytype_values = df['BodyType'].unique()
    socialcontext_values = df['SocialContext'].unique()

    combinations = list(np.array(np.meshgrid(sex_values, bodytype_values, socialcontext_values)).T.reshape(-1, 3))

    # Create a DataFrame for prediction
    data = []
    for combo in combinations:
        row = {
            'Sex': combo[0],
            'BodyType': combo[1],
            'SocialContext': combo[2]
        }
        row.update(fixed_values)
        data.append(row)

    predict_df = pd.DataFrame(data)

    # Match columns to the encoded dataframe structure, ensuring order and missing columns are handled
    predict_encoded = pd.get_dummies(predict_df)
    
    # Select only the columns that were used in the model
    predict_encoded = predict_encoded.reindex(columns=feature_names, fill_value=0)

    # Make predictions
    predictions = model.predict(predict_encoded)
    
    print(predictions)

    # Add predictions to the DataFrame
    predict_df['Predicted_Value'] = predictions

    # Return the best combination with the highest predicted value
    return predict_df.loc[predict_df['Predicted_Value'].idxmax()]


if __name__ == "__main__":
    app.run(debug=True)
