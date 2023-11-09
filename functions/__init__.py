try:
    from .context_spliter import *
    from .embedder import *
    from .pinecone_functions import *
    from .normalizer import *
    import numpy as np
    print("Modules loaded successfully")
except Exception as e:
    print("Loading failed :",e)

def Context_to_Database(context : str,context_id : str,user_id : str) -> bool: 
    """
    1) context를 문장 단위로 나눔.
    2) 각각의 문장을 3개의 임베딩 벡터로 변환.
    3) 벡터들을 데이터 베이스에 업로드.

    Args:
        context (str): 업로드하고자 하는 context
        context_id (str): context의 고유 아이디
        user_id (str): 유저의 고유 아이디
    
    Returns:
        데이터 베이스에 성공적으로 업로드 하였는지 bool type으로 반환.
    """
    #1)
    sentences = split_context_into_sentences(context)
    #2)
    embeddings = sentences_to_embeddings(sentences) 
    lists = []
    for embeddings_per_sentence in embeddings:
        if len(embeddings_per_sentence) >= 1: #한 문장에 대한 여러 임베딩 벡터 중 첫번째, 마지막, 중간 위치의 벡터를 업로드
            lists.append(embeddings_per_sentence[0].tolist())
            lists.append(embeddings_per_sentence[-1].tolist())
            lists.append(embeddings_per_sentence[int(len(embeddings_per_sentence.tolist())/2)].tolist())
    #3)
    return upload_vectors_to_database(lists,context_id,user_id) 

def find_Contexts_related_to_Question(question : str,top_k : int,user_id : str) -> list:
    """
    1) 질의를 임베딩 벡터로 변환.
    2) 질의와 가장 유사한 top_k개의 문장을 데이터베이스에서 선정.
    3) 각각의 문장에 해당하는 context_id를 반환.

    Args:
        question (str): 질의
        top_k (int): 몇 개의 가장 유사한 문장을 선택할지
        user_id (str): 유저의 고유 아이디 (유저가 업로드한 데이터에 대해서만 query를 수행)
    
    Returns:
        context_id(str)가 top_k개 들어있는 list를 반환. (list of string)
    """
    #1)
    Question_vector = sentences_to_embeddings([question])[0][0].tolist()
    #2) 3)
    return Respense_to_context_ids(Query_by_Vector(vector=Question_vector,top_k=top_k,user_id=user_id))

def majority_vote(list_of_context_ids :list) -> str:
    """
    주어진 list_of_context_ids에서 가장 많이 등장하는 요소를 반환.
    find_Contexts_related_to_Question' 함수와 함께 사용.

    Args:
        list_of_context_ids (list): context_id 문자열로 이루어진 리스트

    Returns:
        str: 가장 많이 등장하는 context_id 문자열
    """
    arr = np.array(list_of_context_ids)
    unique_elements, counts = np.unique(arr, return_counts=True)
    sorted_indices = np.argsort(-counts)  
    sorted_unique_elements = unique_elements[sorted_indices]
    return sorted_unique_elements[0]

