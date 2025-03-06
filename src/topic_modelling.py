import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

nltk.download("stopwords")
from nltk.corpus import stopwords

def LDA(text, num_topics=5):
    """Splits text into topic-based sections using LDA."""
    paragraphs = text.split("\n\n") 
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
    X = vectorizer.fit_transform(paragraphs)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_assignments = lda.fit_transform(X).argmax(axis=1)

    sections = {i: [] for i in range(num_topics)}
    for i, topic in enumerate(topic_assignments):
        sections[topic].append(paragraphs[i])

    return ["\n\n".join(sec) for sec in sections.values()]

def transformer_embeddings(text, num_topics=5):
    """
    Splits text into topic-based sections using transformer-based embeddings and KMeans clustering.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(paragraphs)
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    sections = {i: [] for i in range(num_topics)}
    for i, cluster in enumerate(clusters):
        sections[cluster].append(paragraphs[i])
    
    return ["\n\n".join(sec) for sec in sections.values()]

def detect_topics(text, num_topics=5):
    # LDA(text, num_topics)
    transformer_embeddings(text, num_topics)
