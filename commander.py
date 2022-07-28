from model.baseline import Baseline
from model.gpt_neo import GptNeo


class Commander:
    """
    Class designed to communicate with streamlit app
    """
    def __init__(self):
        self.gpt_names = ['gpt_neo_sampled10m_no_hashtags',
                     'gpt_neo_sampled1m',
                     'gpt_neo_sample2m']

        self.baselines_names = [
            'baseline1',
            'baseline2']

        self.all_names = ['gpt_neo_sampled10m_no_hashtags',
                          'gpt_neo_sampled1m',
                          'gpt_neo_sample2m',
                          'baseline1',
                          'baseline2']

        self.prefix = 'model/'
        self.gpt_models = {
                'gpt_neo_sampled10m_no_hashtags':GptNeo(f"{self.prefix}gpt_neo_sampled10m_no_hashtags/model10m"),
                'gpt_neo_sampled1m': GptNeo(f"{self.prefix}gpt_neo_sampled1m/model"),
                'gpt_neo_sample2m': GptNeo(f"{self.prefix}gpt_neo_sample2m/model")
            }

        self.baselines_models = {'baseline1': Baseline(),
                                 'baseline2': Baseline()}

        self.baseline1_root = f"{self.prefix}baseline1"
        self.baseline2_root = f"{self.prefix}baseline2"

    def get(self, model_name):
        if 'gpt in' in model_name:
            return self.gpt_models[model_name]
        if model_name == 'baseline1':
            self.baselines_models[model_name].load(self.baseline1_root)
        else:
            self.baselines_models[model_name].load(self.baseline2_root)
        return self.baselines_models[model_name]

    @property
    def all_names(self):
        return self.all_names

    @all_names.setter
    def all_names(self, value):
        print('Can not set')

