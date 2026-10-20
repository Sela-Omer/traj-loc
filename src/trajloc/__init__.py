"""TrajLoc: trajectory-attention localization for multi-object motion control.

The layers, bottom to top: ``core`` / ``hub``, ``datasets``, ``backbones``,
``training`` / ``inference`` / ``preprocess``, ``generation``, ``cli``; each
package imports only from the layers below it.
"""

__version__ = "1.0.0"
