from happytransformer import GENTrainArgs
from happytransformer import HappyGeneration
from happytransformer import GENSettings
import regex as re


class GptNeo:
    def __init__(self, root):
        self.root = root
        self.happy_gen = HappyGeneration(model_type="GPT-NEO",
                                         model_name="EleutherAI/gpt-neo-125M",
                                         load_path=self.root)

    def load(self):
        return self

    def train(self, path, model_dir):
        """
        trains model with given data and saves it
        :param path: data path
        :param model_dir: desired model save path
        :return:
        """
        gpt_neo = HappyGeneration("GPHappyGeneration, T-Neo", "EleutherAI/gpt-neo-125M")
        train_args = GENTrainArgs(num_train_epochs=1, learning_rate=5e-05, batch_size=2)
        gpt_neo.train(path, args=train_args)
        gpt_neo.save(model_dir)

    def generate(self, start_text, max_length=None):
        """
        generates text sequencing from 'start_text' to prediction
        :param start_text:
        :return:
        """
        top_k_sampling_settings = GENSettings(do_sample=True, max_length=max_length, min_length=1, top_k=50)
        output_top_k_sampling = self.happy_gen.generate_text(start_text, args=top_k_sampling_settings)
        ret = re.sub(r'[0-9]', 'SPLIT', output_top_k_sampling.text).split('SPLIT')
        ret = [r.replace(',', '') for r in ret if r!=""]
        return ret

