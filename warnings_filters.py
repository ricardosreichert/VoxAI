import warnings

# Filtrar avisos de FP16 no Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Filtrar avisos relacionados ao `weights_only=False` no `torch.load`
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False",
    category=FutureWarning,
)

# Filtrar aviso relacionado ao GPT2InferenceModel
warnings.filterwarnings(
    "ignore",
    message=r"GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten.",
    category=FutureWarning,
)

# Filtrar aviso gerado pelo Whisper
warnings.filterwarnings(
    "ignore",
    message=r"FP16 is not supported on CPU",
    category=UserWarning,
)

# Aviso sobre GPT2InferenceModel não herdando `GenerationMixin`
warnings.filterwarnings(
    "ignore",
    message=(
        "GPT2InferenceModel has generative capabilities, as `prepare_inputs_for_generation` "
        "is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`."
    ),
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "From \\\uD83D\\uDC49v4.50\\uD83D\\uDC49 onwards, `PreTrainedModel` will NOT inherit from "
        "`GenerationMixin`, and this model will lose the ability to call `generate` and other related functions."
    ),
    category=FutureWarning,
)

# Outros avisos específicos podem ser adicionados aqui
