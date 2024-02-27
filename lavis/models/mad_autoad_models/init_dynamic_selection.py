from lavis.models.blip2_models.Qformer import BertConfig
from .dynamic_selection  import BertEncoder


def init_dynamic_selection(vision_width, cross_attention_freq, num_hidden_layers):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    bert = BertEncoder(encoder_config)

    return bert