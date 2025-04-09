import torch
import transformers


class LLM:
    def __init__(self, model_name, quantization):
        """Initialise LLM model and quantize for lower memory."""
        self.model_id = model_name
        self.device = (
            f'cuda:{torch.cuda.current_device()}'
            if torch.cuda.is_available()
            else 'cpu'
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantization:
            self.bnb_config = transformers.BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            self.bnb_config = None

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
            device_map=self.device,
        )

        self.model.eval()

    def llama_prompt(self, reference_a, reference_b):
        """Initialise Llama prompt."""

        system = {
            'role': 'system',
            'content': """You are a binary classification model which assesses whether two bibliographic references refer to the same article.

            You will be provided with two bibliographic references and you must classify them as either the Same or Different. Please pay attention to the published year, if these are different, the references
            are most likely different
            You must only respond with the classification and nothing else.
            """,
        }

        assistant = {
            'role': 'assistant',
            'content': """
            Here are some examples of the classification you will be performing:
                Different:
                D Jameson, LM Hurvich, Unknown Author, 1972. . Visual psychophysics. Volume 7.4, Handbook of sensory physiology. , pp. -812.
                Dorothea Jameson, Leo M. Hurvich, 1972. Color Adaptation: Sensitivity, Contrast, After-images. pp. 568-581.
                Same:
                AS Andersen, PH Hansen, L Schäffer, C Kristensen, 2000. A new secreted insect protein belonging to the immunoglobulin superfamily binds insulin and related peptides and inhibits their activities. J Biol Chem. 275, pp. 16948-16953.
                Asser Sloth Andersen, Per Hertz Hansen, Lauge Schäffer, Claus Kristensen, 2000. A New Secreted Insect Protein Belonging to the Immunoglobulin Superfamily Binds Insulin and Related Peptides and Inhibits Their Activities. Journal of Biological Chemistry. 275, pp. 16948-16953.
                Same:
                AL Fisher, S Ohsako, M Caudy, 1996. The WRPW motif of the Hairy-related basic helix-loop-helix repressor proteins acts as a 4-amino acid transcription repression and protein–protein interaction domain. Mol Cell Biol. 16, pp. 2670-2677.
                Alfred L. Fisher, Shunji Ohsako, Michael Caudy, 1996. The WRPW Motif of the Hairy-Related Basic Helix-Loop-Helix Repressor Proteins Acts as a 4-Amino-Acid Transcription Repression and Protein-Protein Interaction Domain. Molecular and Cellular Biology. 16, pp. 2670-2677.
                Different:
                E Abouheif, 1999. A method for testing the assumption of phylogenetic independence in comparative data. Evol Ecol Res. 1, pp. 895-909.
                Yuki Haba, Nobuyuki Kutsukake, 2019. A multivariate phylogenetic comparative method incorporating a flexible function between discrete and continuous traits. Evol Ecol. 33, pp. 751-768.
                Different:
                L Medway, 1977, Mammals of Borneo, Monogr Malay Br R Asia Soc, 7, 1-172, None, None
                G. S. Jones, 1981. Lord Medway. MAMMALS OF BORNEO: FIELD KEYS AND AN ANNOTATED CHECKLIST. 2nd edition. Monogr. Malay. Br. Roy. Asiat. Soc., 7:xii + 172 pp. 1977. Journal of Mammalogy. 62, pp. 217-218.
            """,
        }

        user = {
            'role': 'user',
            'content': f"""
                These are the two references {reference_a}, {reference_b}.
                Do they refer to the same article? Please respond with 'Same' or 'Different'.
                """,
        }

        return [system, assistant, user]

    def instruct_model(self, reference_a, reference_b):
        """Retrieve generated text from model."""

        encoded_inputs = self.tokenizer.apply_chat_template(
            self.llama_prompt(reference_a=reference_a, reference_b=reference_b),
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )

        if isinstance(encoded_inputs, dict):
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
        elif isinstance(encoded_inputs, torch.Tensor):
            input_ids = encoded_inputs.to(self.device)
            attention_mask = None 
        else:
            raise ValueError("Unexpected output from apply_chat_template")

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
        ]

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = outputs[:, input_ids.shape[-1]:]

        return self.tokenizer.decode(response[0], skip_special_tokens=True)
