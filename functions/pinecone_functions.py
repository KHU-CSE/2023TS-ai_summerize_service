import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(
    api_key=api_key,
    environment="gcp-starter" )
index_name = "ask-everything" 
index = pinecone.Index(index_name) #client instance


def upload_vectors_to_database(list_of_vectors : list, context_id : str, user_id : str) -> bool:
    try:
        temp = [(context_id+"_"+str(i), vector , {"context_id":context_id, "user_id" : user_id}) for i,vector in enumerate(list_of_vectors)]
        response = index.upsert(temp)
        return True
    except:
        return False

def Query_by_Vector(vector : list,top_k : int,user_id : str) -> dict:
    query_response = index.query(
                        vector= vector,
                        top_k= top_k,
                        include_values= True,
                        include_metadata= True,
                        filter= {"user_id" : {"$eq" : user_id}}
                    )
    return query_response

def Respense_to_context_ids(response):
    temp = []
    for result in response["matches"]:
        temp.append(result["metadata"]["context_id"])
    return temp

def reset_database():
    if input("really?") == "YeS":
        pinecone.delete_index(index_name)
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric='cosine'
        )
