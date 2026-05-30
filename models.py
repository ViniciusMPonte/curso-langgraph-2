from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY')

_PROVIDER_MAP: dict[str, type[BaseChatModel]] = {
    'google': ChatGoogleGenerativeAI
}

MODEL_CONFIGS: list[dict] = [
    {
        'key_name': 'gemini-2.5-flash-lite',
        'provider': 'google',
        'model_name': 'gemini-2.5-flash-lite',
        'temperature': 0.2,
    },
    {
        'key_name': 'gemini-3.1-flash-lite',
        'provider': 'google',
        'model_name': 'gemini-3.1-flash-lite',
        'temperature': 0.2,
    }
]

def _create_chat_model(model_name: str, provider: str, temperature: float | None = None) ->  BaseChatModel:
    if provider not in _PROVIDER_MAP:
        raise ValueError(f'Provedor não suportado: {provider}. Provedores suportados são: {list(_PROVIDER_MAP.keys())}')

    model_class: BaseChatModel = _PROVIDER_MAP[provider]
    params: dict = {'model': model_name}

    if temperature is not None:
        params['temperature'] = temperature

    return model_class(**params)

models: dict[str, BaseChatModel] = {}

for config in MODEL_CONFIGS:
    models[config['key_name']] = _create_chat_model(
        model_name=config['model_name'],
        provider=config['provider'],
        temperature=config.get('temperature')
    )