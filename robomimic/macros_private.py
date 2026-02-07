"""
Set of global variables shared across robomimic
"""
# Sets debugging mode. Should be set at top-level script so that internal
# debugging functionalities are made active
DEBUG = False

# Whether to visualize the before & after of an observation randomizer
VISUALIZE_RANDOMIZER = False

# wandb entity (eg. username or team name)
WANDB_ENTITY = 'zillur'

# wandb api key (obtain from https://wandb.ai/authorize)
# alternatively, set up wandb from terminal with `wandb login`
WANDB_API_KEY = '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'

# Key in obs dict used for CLIP language embeddings
LANG_EMB_KEY = "lang_emb"
LANG_KEY = "lang"

assert LANG_KEY in ["lang_var", "lang"]
try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import robomimic
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
    )
