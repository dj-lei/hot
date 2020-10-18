import pandas as pd
import argparse
from model.models import *
from db.db import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis_config", type=str, default="172.17.0.2:6379:1", help="redis config")
    parser.add_argument("--mysql_config", type=str, default="172.17.0.3:root:123456:hot", help="mysql config")
    opt = parser.parse_args()

    redis_ctrl = RedisCtrl(opt.redis_config.split(':'))
    mysql_ctrl = MysqlCtrl(opt.mysql_config.split(':'))

    image_net = ResidualNet()
    image_net.eval()
    image_net = image_net.cuda()

    text_net = RoBertaChinese()

    with torch.no_grad():
        best_seller = mysql_ctrl.load('best_seller')
        image_vectors = []
        text_vectors = []
        for i in range(0, len(best_seller)):
            image_path = 'data/best_seller/' + best_seller['id'][i] + '.jpg'
            image_space_vector = get_image_space_vector(image_path, image_net, 'avg')
            text_space_vector = get_text_space_vector(best_seller['title'][i], text_net.tokenizer,
                                                      text_net.model).tolist()
            image_vectors.append(image_space_vector)
            text_vectors.append(text_space_vector)
        best_seller['image_vectors'] = pd.Series(image_vectors)
        best_seller['text_vectors'] = pd.Series(text_vectors)

    with torch.no_grad():
        source_goods = mysql_ctrl.load('source_goods')
        image_vectors = []
        text_vectors = []
        for i in range(0, len(source_goods)):
            image_path = 'data/source_goods/' + source_goods['id'][i] + '.jpg'
            image_space_vector = get_image_space_vector(image_path, image_net, 'avg')
            text_space_vector = get_text_space_vector(source_goods['title'][i], text_net.tokenizer,
                                                      text_net.model).tolist()
            image_vectors.append(image_space_vector)
            text_vectors.append(text_space_vector)
        source_goods['image_vectors'] = pd.Series(image_vectors)
        source_goods['text_vectors'] = pd.Series(text_vectors)

    source_goods['image_mean_cosine'] = 0
    list_best_seller_vectors = best_seller.image_vectors.values
    for i in range(0, len(source_goods)):
        mean_cosine = get_good_mean_cosine(source_goods['image_vectors'][i], list_best_seller_vectors)
        source_goods.loc[i, 'image_mean_cosine'] = mean_cosine

    source_goods['text_mean_cosine'] = 0
    list_best_seller_vectors = best_seller.text_vectors.values
    for i in range(0, len(source_goods)):
        mean_cosine = get_good_mean_cosine(source_goods['text_vectors'][i], list_best_seller_vectors)
        source_goods.loc[i, 'text_mean_cosine'] = mean_cosine