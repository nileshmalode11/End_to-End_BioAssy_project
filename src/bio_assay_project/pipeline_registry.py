"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from typing import Dict

from kedro.pipeline import Pipeline

from bio_assay_project.pipelines import data_processing as dp
from bio_assay_project.pipelines import data_science as ds
# from bioassay_project.pipelines import inference as infer


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    # inference_pipeline=infer.create_pipeline()

    return {
        "__default__": data_processing_pipeline + data_science_pipeline, #+ inference_pipeline,
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        # "infer":inference_pipeline
    }