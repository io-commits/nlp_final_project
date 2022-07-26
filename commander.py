from model.baseline import Baseline
from model.gpt_neo import GptNeo
import pandas as pd


class Commander:
    """
    Class designed to communicate with streamlit app
    It is in charge of initializing the models with proper values
    """
    def __init__(self):
        self.gpt_names = ['gpt_neo_sampled30m_no_hashtags',
                          'gpt_neo_sampled10m_no_hashtags',
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
                'gpt_neo_sampled30m_no_hashtags': GptNeo(f"{self.prefix}gpt_neo_sampled30m_no_hashtags/model30m"),
                'gpt_neo_sampled10m_no_hashtags':GptNeo(f"{self.prefix}gpt_neo_sampled10m_no_hashtags/model10m"),
                'gpt_neo_sampled1m': GptNeo(f"{self.prefix}gpt_neo_sampled1m/model"),
                'gpt_neo_sample2m': GptNeo(f"{self.prefix}gpt_neo_sample2m/model")
            }

        self.baseline1_root = f"{self.prefix}baseline1"
        self.baseline2_root = f"{self.prefix}baseline2"

        # self.baselines_models = {'baseline1': Baseline(self.baseline1_root),
        #                          'baseline2': Baseline(self.baseline2_root)}
        self.test_df = pd.read_csv('data/test.csv', low_memory=False, lineterminator='\n', index_col=0)

    def get(self, model_name):
        """
        returns actual model by it name
        :param model_name: model's name
        :return: model object
        """
        if 'gpt' in model_name:
            return self.gpt_models[model_name]

        # if model_name == 'baseline1':
        #     self.baselines_models[model_name].load(self.baseline1_root)
        # else:
        #     self.baselines_models[model_name].load(self.baseline2_root)
        #
        # return self.baselines_models[model_name]

    def sample(self):
        """
        samples the test df
        :return: sampled DataFrame with one row
        """
        return self.test_df.loc[:, 'text'].sample().values[0]



