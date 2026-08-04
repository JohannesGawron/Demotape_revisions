'''This project is supposed to address concerns that came up in the revisions of DemoTape regarding the claim that clonal structures and evolutionary patterns inferred from multiplexed data closely match those from individually sequenced samples. The authors report noticeable discrepancies in SNP detection between demultiplexed and individually sequenced samples, which may indicate misidentified mutations that in turn may affect downstream clonal inference. While some differences are attributed to filtering thresholds by the authors, here we show that these discrepancies do not introduce systematic biases in genotype reconstruction.
 (July 28, 2026).

'''

from mutation_call_variability._version import __version__

__all__ = ["__version__"]
