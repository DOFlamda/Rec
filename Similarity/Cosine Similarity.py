#cosine_similarity(A,B)=(A dot B)/(||A||*||B||)
#两向量内积与两个向量的模长的乘积之比
#=1,向量A和B方向相同,完全相似
#=0,向量A和B正交，垂直
#=-1,向量A和B方向相反，完全不相似

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例文本
text1 = "This is the first document."
text2 = "This document is the second document."
text3 = "And this is the third one."
text4 = "Is this the first document?"

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2, text3, text4])# 将文本向量化
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 打印相似性矩阵
print("相似性矩阵：")
print(cosine_sim)

print("\n文本之间的相似性：")
print("文本1与文本2的相似性:", cosine_sim[0][1])
print("文本1与文本3的相似性:", cosine_sim[0][2])
print("文本1与文本4的相似性:", cosine_sim[0][3])
