import nltk

from services import text_processing


def download_nltk_data():
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def get_embedding_service(embedding):
    if embedding is None:
        from models import database
        from services import embeddings

        embedding = embeddings.Embeddings(session=database.session)

    return embedding


def get_wordnet_syn_service(wordnet_syn):
    download_nltk_data()

    if wordnet_syn is None:
        from services import embeddings

        wordnet_syn = embeddings.WordnetSyn("por")

    return wordnet_syn



def initialize():
    from models import database

    database.Base.metadata.create_all(bind=database.engine)

    session = database.LocalSession()
    data = text_processing.parse_pdfs(session=session)
    
    embedding = get_embedding_service(None)
    
    for key in data.keys():
        embedding.process_data(data[key])

    return embedding

    # wordnet_syn._precompute_mapping()
    # faiss_index.build_index()
    # graph.build_graph_network()

    
        

