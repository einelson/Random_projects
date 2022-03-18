from data.bee_movie import script
bee_list = script.split('\n')
bee_list = list(filter(None, bee_list))
train_sentence = [ele for ele in bee_list if ele.strip()]