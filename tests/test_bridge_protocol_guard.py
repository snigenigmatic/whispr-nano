"""The retired fixed-split path must fail before any experiment or spend mutation."""
from unittest.mock import patch
import pytest
from research.bridge_open import launch_fleurs_shard


def test_legacy_long_dispatch_requires_explicit_diagnostic_override():
    args=['launch_fleurs_shard','--model','turbo','--language','hi','--route','long']
    with patch('sys.argv',args), patch.object(launch_fleurs_shard,'command') as command:
        with pytest.raises(ValueError,match='retired'):
            launch_fleurs_shard.main()
        command.assert_not_called()
