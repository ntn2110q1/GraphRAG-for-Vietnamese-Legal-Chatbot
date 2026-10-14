from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'legal_qa.db').as_posix()}"
    legal_data_dir: str = str(PROJECT_ROOT / "data")
    admin_api_token: str = ""
    legal_max_upload_mb: int = 20
    legal_chunk_size: int = 1400
    legal_chunk_overlap: int = 180
    legal_top_k: int = 5
    legal_llm_evidence_k: int = 3
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    legal_llm_backend: str = "ollama"
    legal_hf_model: str = ""
    legal_adapter_path: str = ""
    legal_hf_4bit: bool = False
    legal_embedding_model: str = ""
    legal_eval_dataset: str = str(PROJECT_ROOT / "evaluation" / "datasets" / "eval.jsonl")

    @property
    def data_path(self) -> Path:
        path = Path(self.legal_data_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


settings = Settings()
