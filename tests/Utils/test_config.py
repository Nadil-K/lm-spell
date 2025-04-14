from unittest.mock import mock_open, patch
from Utils.Config import Config
import json

def test_get_configs():

    mock_config_data = json.dumps({
        "dataset": {
            "seq2seq": {
                "inputs": "text",
                "targets": "expected"
            }
        }
    })
    
    with patch("builtins.open", mock_open(read_data=mock_config_data)), patch("os.path.exists", return_value=True):

        config = Config()
        assert config.get('dataset.seq2seq.inputs') == 'text'
        assert config.get('dataset.seq2seq.targets') == 'expected'
        assert config.get('dataset.seq2seq.non_existing_key') is None
        assert config.get('non_existing_key', 'default_value') == 'default_value'