import numpy as np

user_ratings ={
    'User1': {'Item1': 5, 'Item2': 3, 'Item3': 4, 'Item4': 3, 'Item5': 1},
    'User2': {'Item1': 3, 'Item2': 1, 'Item3': 3, 'Item4': 3, 'Item5': 5},
    'User3': {'Item1': 4, 'Item2': 2, 'Item3': 4, 'Item4': 1, 'Item5': 5},
    'User4': {'Item1': 4, 'Item2': 3, 'Item3': 3, 'Item4': 5, 'Item5': 2},
    'User5': {'Item1': 3, 'Item2': 5, 'Item4': 4, 'Item5': 1},
}

def cosine_similarity(user1, user2):
    common_items = set(user1.keys()) & set(user2.keys())
    if len(common_items) == 0:
        return 0.0
    vector1 = [user1[item] for item in common_items]
    vector2 = [user2[item] for item in common_items]
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def user_based_recommendation(target_user, user_ratings):
    # 计算目标用户与其他用户的相似性
    similarities = {}
    for user, ratings in user_ratings.items():
        if user != target_user:
            similarity = cosine_similarity(user_ratings[target_user], ratings)
            similarities[user] = similarity  #其他用户的值为目标用户的物品评分与其他用户的评分相似度

    # 推荐未被目标用户评分的物品
    recommendations = {}
    for item in user_ratings[target_user]:
        if user_ratings[target_user][item] == 0:  # 用户未评分的物品
            weighted_sum = 0
            similarity_sum = 0
            for user, similarity in similarities.items():
                if user_ratings[user][item] > 0:  # 用户已评分的物品
                    weighted_sum += user_ratings[user][item] * similarity
                    similarity_sum += abs(similarity)
            if similarity_sum > 0:
                recommendations[item] = weighted_sum / similarity_sum
    # 按推荐分数降序排序
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations

# 为指定用户生成推荐
target_user = 'User5'
recommendations = user_based_recommendation(target_user, user_ratings)

# 打印推荐结果
print(f"为用户 {target_user} 生成的推荐物品：")
for item, score in recommendations:
    print(f"{item}: 推荐分数 {score}")
