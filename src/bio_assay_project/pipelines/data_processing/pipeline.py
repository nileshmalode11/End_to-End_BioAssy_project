from kedro.pipeline import Pipeline,node

from .nodes import concat,df_bio_shape,df_bio_count,df_bio_split,df_bio_balance,df_bio_scaling,df_bio_pca,df_bio_pca_100

def create_pipeline(**kwargs):
    return Pipeline(
        [   
            node(
                func=concat,
                inputs=["df746_1284_train","df746_1284_test"],
                outputs="df_bio",
                name="concat",
            ),
            node(
                func=df_bio_shape,
                inputs="df_bio",
                outputs="df_bio_shape",
                name="df_bio_shape",
            ),
            node(
                func=df_bio_count,
                inputs="df_bio",
                outputs="df_bio_count",
                name="df_bio_count",

            ),
             node(
                func=df_bio_split,
                inputs="df_bio",
                outputs=["X","y"],
                name="df_bio_split",
            ),
            node(
                func=df_bio_balance,
                inputs=["X","y"],
                outputs=["X_sample","y_sample"],
                name="df_bio_balance",
            ),
            node(
                func=df_bio_scaling,
                inputs="X_sample",
                outputs="scaling_data",
                name="df_bio_scaling",
            ),
            node( 
                func=df_bio_pca,
                inputs="scaling_data",
                outputs="X_pca",
                name="df_bio_pca",
            ),
            node(
                func=df_bio_pca_100,
                inputs="scaling_data",
                outputs="X_pca_100",
                name="df_bio_pca_100",
            )
                
        ]
    ) 