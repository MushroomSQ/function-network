from config.config_pqnet import PQNetConfig


def get_config(name):
    if name == 'pqnet':
        return PQNetConfig
    else:
        raise ValueError("Got name: {}".format(name))
