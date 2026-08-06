"""Version information for cell_analysis.

This file is automatically updated by hatch-vcs when building from git tags.
For development installs without git tags, a fallback version is provided.
"""

try:
    from mutation_call_variability._version_generated import __version__
except ImportError:
    # Fallback for editable installs or when not built with hatch-vcs
    __version__ = '0.1.0.dev0'
