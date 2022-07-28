from happytransformer import HappyGeneration, GENTrainArgs
from happytransformer import HappyGeneration
from happytransformer import GENSettings


class GptNeo:
    def __init__(self, root):
        self.root = root
        self.happy_gen = None

    def load(self):
        """
        initializes the happy transformer object from the root directory.
        :param root: location to load the model from
        """
        self.happy_gen = HappyGeneration(model_type="GPT-NEO",
                                         model_name="EleutherAI/gpt-neo-125M",
                                         load_path=self.root)

    def train(self, path, model_dir):
        """
        trains model with given data and saves it
        :param path: data path
        :param model_dir: desired model save path
        :return:
        """
        gpt_neo = HappyGeneration("GPT-Neo", "EleutherAI/gpt-neo-125M")
        train_args = GENTrainArgs(num_train_epochs=1, learning_rate=5e-05, batch_size=2)
        gpt_neo.train(path, args=train_args)
        gpt_neo.save(model_dir)

    def generate(self, start_text):
        """
        generates text sequencing from 'start_text' to prediction
        :param start_text:
        :return:
        """
        top_k_sampling_settings = GENSettings(do_sample=True, top_k=50, max_length=30, min_length=10)
        output_top_k_sampling = self.happy_gen.generate_text("peace", args=top_k_sampling_settings)
        return output_top_k_sampling.text



