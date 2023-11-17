from flask import Flask,render_template,request
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# from pandas.core.indexes.numeric import NumericIndex



popular_df = pickle.load(open('final_products.pkl','rb'))
cp = pickle.load(open('cleaned_prod.pkl','rb'))
cp2 = pickle.load(open('final_prod2.pkl','rb'))
tfv_matrix =  pickle.load(open('similarity.pkl','rb'))
tfv = TfidfVectorizer(ngram_range=(1,2))
final_rating_matrix = pickle.load(open('Final_rat_matr.pkl','rb'))
pred_matrix = pickle.load(open('pred_matrix.pkl','rb'))
sparse_matrix = pickle.load(open('sparse_matrix.pkl','rb'))

def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(0, interactions_matrix.shape[0]): #  .shape[0] gives number of rows
        
        #finding cosine similarity between the user_id and each user
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        
        #Appending the user and the corresponding similarity score with user_id as a tuple
        similarity.append((user,sim))
        
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity] #Extract the user from each tuple in the sorted list
    similarity_score = [tup[1] for tup in similarity] ##Extracting the similarity score from each tuple in the sorted list
   
    #Remove the original user and its similarity score and keep only other similar users 
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])
       
    return most_similar_users, similarity_score
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',product_name=list(popular_df['product_name'].values),
                            retail_price=list(popular_df['retail_price'].values),
                            discounted_price=list(popular_df['discounted_price'].values),
                            image=list(popular_df['image'].values),
                                             )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_product',methods = ['post'])
def recommend():
    user_input = request.form.get('user_input')
    tfv_matrix = tfv.fit_transform(cp['tags'])
    query_vec = tfv.transform([user_input])
    similarity = cosine_similarity(query_vec,tfv_matrix).flatten()
    
    indices = np.argpartition(similarity,-20)[-20:]
    results = cp.iloc[indices]
    
    data=results.values.tolist() 
    
    return render_template('recommend.html',data = data)



#********************************************************************


@app.route('/simiprod')
def simiprod():
    return render_template('SimiCollab.html')

@app.route('/Simi_prod',methods = ['post'])
def recommendations():
    user_index = int(request.form.get('user_input'))
    #Saving similar users using the function similar_users defined above
    most_similar_users = similar_users(user_index, final_rating_matrix)[0]
    
    #Finding product IDs with which the user_id has interacted
    prod_ids = set(list(final_rating_matrix.columns[np.where(final_rating_matrix.loc[user_index] > 0)]))
    recommendations = []
    
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < 20:
            
            #Finding 'n' products which have been rated by similar users but not by the user_id
            similar_user_prod_ids = set(list(final_rating_matrix.columns[np.where(final_rating_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    
    d =  recommendations[:20]

    data=cp2[cp2.isin (d).any (1)].values.tolist()
        
    return render_template('SimiCollab.html',data = data)

#**********************************************************************************************************










                           
if __name__ == "__main__":
    app.run(debug = True)