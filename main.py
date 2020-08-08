"""
main.py - Only this file is intended to be run.
"""
import os

from tensorflow import keras


from build_configuration import BuildConfiguration
from text_processor import TextProcessor
from model_definition import ModelDefinition
from datasets import Datasets
from build_summary import BuildSummary
from model_evaluation import ModelEvaluation


if BuildConfiguration.CREATE_DATASET_VISUALS:
    Datasets.create_intents_bar_plots()
    Datasets.create_intents_venn_diagram()

data = TextProcessor(
    Datasets.training_data,
    Datasets.testing_data,
    ModelDefinition.tokenizer,
    Datasets.all_intents
)

print("train x shape: ", data.train_x.shape)
print("train x element example: ", data.train_x[0])
print("train y element example: ", data.train_y[0])
print("data max_length_sequence", data.max_sequence_length)

bert_model = ModelDefinition.create_model(data.max_sequence_length, len(Datasets.all_intents))
print(bert_model.summary())

# TODO - Open a number of options for optimizers, loss functions, and metrics.
bert_model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=BuildConfiguration.LOG_DIR)

build_history = bert_model.fit(
    x=data.train_x,
    y=data.train_y,
    validation_split=BuildConfiguration.VALIDATION_SPLIT,
    batch_size=BuildConfiguration.BATCH_SIZE,
    epochs=BuildConfiguration.EPOCHS,
    callbacks=[tensorboard_callback]
)

if BuildConfiguration.CREATE_BUILD_VISUALS:
    BuildSummary.create_accuracy_over_epochs_plot(build_history=build_history)
    BuildSummary.create_loss_over_epochs_plot(build_history=build_history)

ModelEvaluation.evaluate_model_accuracy(bert_model=bert_model, data=data)
ModelEvaluation.create_classification_report(bert_model=bert_model, data=data)
ModelEvaluation.create_confusion_matrix(bert_model=bert_model, data=data)
ModelEvaluation.perform_regression_testing(bert_model=bert_model, data=data)

print("SAVING BERT MODEL")
saved_model_path = os.path.join(BuildConfiguration.OUTPUT_DIR, "model")
keras.models.save_model(bert_model, saved_model_path)

# print("TESTING BERT MODEL LOAD")
# loaded_bert_model = keras.models.load_model(saved_model_path)
#
# loaded_predictions = loaded_bert_model.predict(pred_token_ids).argmax(axis=-1)
#
# for utterance, intent in zip(sentences, loaded_predictions):
#     print(" utterance:", utterance, "\n intent:", Datasets.all_intents[intent])
