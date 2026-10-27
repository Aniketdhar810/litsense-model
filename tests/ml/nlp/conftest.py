"""Reuse the data tests' fixtures (tiny fixture over a local HTTP server) and register the stages
the pipeline test runs."""

import ml.data.canonical  # noqa: F401  (registers the stage)
import ml.data.clean  # noqa: F401  (registers the stage)
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.eda  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
import ml.data.reviews  # noqa: F401  (registers the stage)
import ml.data.split  # noqa: F401  (registers the stage)
import ml.nlp.annotate  # noqa: F401  (registers the stage)
import ml.nlp.aspects  # noqa: F401  (registers the stage)
import ml.nlp.centroids  # noqa: F401  (registers the stages)
import ml.nlp.profiles_offline  # noqa: F401  (registers the stages)
import ml.nlp.sentences  # noqa: F401  (registers the stage)
import ml.nlp.sentiment  # noqa: F401  (registers the stage)
import ml.nlp.snippets  # noqa: F401  (registers the stage)
import ml.nlp.spoiler_train  # noqa: F401  (registers the stages)
from tests.ml.data.conftest import file_server, tiny_cfg  # noqa: F401
