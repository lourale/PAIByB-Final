import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FeatureEvaluator:
    def __init__(self, df, analysis = False):
        """
        Recibe un DataFrame con características extraídas y una columna 'label' con las etiquetas.
        """
        self.df = df
        if analysis:
            self.df_ranking = self.__rank_features()
            self.__plot_correlation_matrix()
        
    def __rank_features(self):
        """
        Evalúa la importancia de cada característica usando Información Mutua 
        y un mini-modelo de Random Forest.
        """
        
        # Separamos características (X) y etiquetas (y)
        X = self.df.drop('label', axis=1)
        y = self.df['label']

        # 1. Prueba Estadística: Información Mutua
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # 2. Prueba Mini-Modelo: Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_

        # 3. Consolidar en un DataFrame
        ranking = pd.DataFrame({
            'Caracteristica': X.columns,
            'Info_Mutua': mi_scores,
            'RF_Importancia': rf_scores
        })

        # Normalizamos ambos puntajes de 0 a 1 para poder promediarlos justamente
        ranking['MI_Norm'] = ranking['Info_Mutua'] / ranking['Info_Mutua'].max()
        ranking['RF_Norm'] = ranking['RF_Importancia'] / ranking['RF_Importancia'].max()
        
        # Calculamos un Score Final Promedio
        ranking['Score_Final'] = (ranking['MI_Norm'] + ranking['RF_Norm']) / 2

        # Ordenamos de mejor a peor
        ranking = ranking.sort_values(by='Score_Final', ascending=False).reset_index(drop=True)

        print("\n--- RANKING DE CARACTERÍSTICAS ---")
        print(ranking[['Caracteristica', 'Score_Final', 'Info_Mutua', 'RF_Importancia']])
        
        # Ploteo rápido del ranking
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score_Final', y='Caracteristica', data=ranking, palette='viridis')
        plt.title('Importancia Global de Características (Mutual Info + Random Forest)')
        plt.xlabel('Score Combinado (Normalizado 0-1)')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

        return ranking

    def __plot_correlation_matrix(self):
        """
        Genera un Heatmap de correlación de Pearson para detectar 
        características redundantes (clones matemáticos).
        """
        print("Calculando matriz de correlación cruzada...")
        
        # Quitamos la etiqueta de texto para poder hacer matemáticas
        df_numeric = self.df.drop('label', axis=1)
        
        # Calculamos la correlación
        corr_matrix = df_numeric.corr()
        
        # Configuramos el lienzo
        plt.figure(figsize=(14, 10))
        plt.title("Matriz de Correlación de Características (Buscando Redundancias)", fontsize=14)
        
        # Ploteamos el heatmap. 
        # vmin=-1 y vmax=1 son los límites matemáticos.
        # annot=True escribe el numerito adentro de cada cuadro.
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, square=True, linewidths=.5)
        
        plt.tight_layout()
        plt.show()

    def run_models_benchmark(self, elite_features):
        """Entrena y evalúa 4 modelos distintos usando solo el "Dream Team" de características.
        """
        print("-" * 50)
        print("INICIANDO BENCHMARK DE MODELOS")
        print("-" * 50)
        
        # Verificamos que existan en el df, si no, usamos las que haya
        features_to_use = [f for f in elite_features if f in self.df.columns]
        
        X = self.df[features_to_use]
        y = self.df['label']
        
        # Partimos los datos: 80% para entrenar, 20% para examen
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Datos de Entrenamiento: {X_train.shape[0]} parches")
        print(f"Datos de Prueba (Test): {X_test.shape[0]} parches\n")
        
        # Definimos los competidores
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting (XGBoost lite)": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
            "Naïve Bayes": GaussianNB()
        }
        
        # Entrenamos, evaluamos y mostramos resultados
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
            
            print(f"--- {name} ---")
            print(f"Precisión Global (Accuracy): {acc * 100:.2f}%")
            print("." * 30)
            
        best_model_name = max(results, key=results.get)
        print(f"\nGANADOR DEL BENCHMARK: {best_model_name} con {results[best_model_name] * 100:.2f}% de precisión.")
        
        return models # Retornamos los modelos entrenados por si queremos usarlos luego

    def rank_feature_combinations(self, elite_features, max_size=None):
        """
        Prueba diferentes combinaciones de features y las rankea por Accuracy.
        max_size: El número máximo de features por combinación (para evitar que tarde años).
        """
        if max_size is None:
            max_size = len(elite_features)
        
        results = []
        y = self.df['label']
        
        print(f"Analizando combinaciones de '{elite_features}'...")

        # Generamos todas las combinaciones posibles desde tamaño 1 hasta max_size
        for r in range(1, max_size + 1):
            for combo in combinations(elite_features, r):
                combo = list(combo)
                X = self.df[combo]
                
                # Split rápido para evaluar
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Entrenamos un RF rápido (pocos estimadores para velocidad)
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                results.append({
                    'num_features': len(combo),
                    'features': ", ".join(combo),
                    'accuracy': acc
                })

        # Crear DataFrame de ranking
        ranking_df = pd.DataFrame(results).sort_values(by='accuracy', ascending=False)
        
        print("\n--- TOP 5 MEJORES COMBINACIONES ---")
        #imprimir cada feature por separado para mejor legibilidad
        print(ranking_df.head(10))
        
        print("\n--- MEJOR COMBINACIÓN COMPLETA --- ")
        best_row = ranking_df.iloc[0]
        print("Features del mejor ranking:")
        print(best_row['features'])
        
        return ranking_df    
        
