import redis
from sqlalchemy import create_engine
import pandas as pd


class RedisCtrl(object):
    def __init__(self, config):
        pool = redis.ConnectionPool(host=config[0], port=config[1], decode_responses=True, db=config[2])
        self.redis_con = redis.Redis(connection_pool=pool)

    def save_key_values(self, key, values):
        self.redis_con.flushdb()
        for value in values:
            self.redis_con.sadd(key, str(value))
        return True

    def get_key_number(self, key):
        return self.redis_con.scard(key)


class MysqlCtrl(object):
    def __init__(self, config):
        mysql_engin = 'mysql+pymysql://' + config[1] + ':' + config[2] + '@' + config[0] + '/' + config[3] + '?charset=utf8'
        self.engine = create_engine(mysql_engin)

    def save(self, df, table):
        df.to_sql(table, con=self.engine, if_exists='append', index=False)

    def load(self, table):
        sql = 'select * from ' + table + ';'
        return pd.read_sql_query(sql, self.engine)
