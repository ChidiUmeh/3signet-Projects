import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

df =pd.read_csv('cleaned_data.csv')
# Feature creation
def create(df, col1, col2, new_col):
    df[new_col] = df[col1] + df[col2]
    return df[new_col]

# Feature interaction
def plot(df):
    fig =  plt.figure()
    sns.pairplot(df)
    return fig
f = plot(df)
plt.show()

# Polynomial Features
def poly(df, num):
    poly = PolynomialFeatures(degree=3,interaction_only=True)
    return poly.fit_transform(df[num].values)

# Feature Transformation
def transform(df, num):
    for n in num:
        df[n]=np.log10(df[n]+1)
        return df[n]

def bin(df, num, bins, labels):
    df[num] = pd.cut(df[num],bins=bins, labels=labels)
    return df[num]

def scale(df, num):
    scaler = MinMaxScaler()
    df[num] = scaler.fit_transform(df[num])
    return df[num]

# Feature selection
def select_feature1(df, target):
    X_columns = df.drop(target, axis=1).columns
    X= df.drop(target,axis=1).values
    y= df[target].values
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, y)
    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features = fit.transform(X)
    # Summarize selected features
    return features[0:5,:]

# Feature extraction
def select_feature2(df, target):
    X_columns = df.drop(target, axis=1).columns
    X= df.drop(target,axis=1).values
    y= df[target].values
    model = DecisionTreeClassifier()
    rfe = RFE(estimator=model, n_features_to_select=10)
    fit = rfe.fit(X, y)
    important_features = pd.DataFrame(list(zip(X_columns,fit.support_)),columns=['Feature','Important'])
    return important_features.sort_values(by='Important', ascending=False)

def select_feature3(df, target):
    X_columns = df.drop(target, axis=1).columns
    X= df.drop(target,axis=1).values
    y= df[target].values
    ridge = Ridge(alpha=1.0)
    ridge.fit(X,y)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, random_state=None, solver='auto', tol=0.001)
    # A helper method for pretty-printing the coefficients
    def pretty_print_coefs(coefs, names = None, sort = False):
        if names == None:
            names = ["X%s" % x for x in range(len(coefs))]
            lst = zip(coefs, names)
            if sort:
               lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
    print ("Ridge model:", pretty_print_coefs(ridge.coef_))

def select_feature4(df, target):
    from sklearn.decomposition import PCA
    X_columns = df.drop(target, axis=1).columns
    X= df.drop(target,axis=1).values
    y= df[target].values
    pca = PCA(10)
    pca.fit_transform(X)
    return pca.n_components_, pca.explained_variance_ratio_

def select_feature5(df, target):
    X_columns = df.drop(target, axis=1).columns
    X= df.drop(target,axis=1).values
    y= df[target].values
    tsne =TSNE(n_components=2,perplexity=20,random_state=42)
    tsne_df = tsne.fit_transform(X)
    fig = plt.figure(figsize=(10,10))
    plt.scatter(tsne_df[:,0],tsne_df[:,1],c=y)
    plt.legend()
    return fig


