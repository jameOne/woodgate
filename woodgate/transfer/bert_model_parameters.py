"""
bert_model_parameters.py - This module contains the
BertModelParameters class definition
"""
import os
from typing import List, Union


class BertModelParameters:
    """
    BertModelParameters - The BertModelParameters class
    encapsulates logic related to defining the specific BERT
    model used for transfer learning.
    """

    #: The `ALLOWED_L_VALUES` attribute is a constant which
    #: represents the number of stacked encoders in BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_L_VALUES: List[int] = [2, 4, 6, 8, 10, 12, 24]

    #: The `ALLOWED_H_VALUES` attribute is a constant which
    #: represents the number of hidden size of the BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_H_VALUES: List[int] = [128, 256, 512, 768, 1024]

    #: The `ALLOWED_A_VALUES` attribute is a constant which
    #: represents the number of head in the attention layers of
    #: the BERT model.
    #:
    #: See `https://github.com/google-research/bert` for more on
    #: BERT.
    ALLOWED_A_VALUES: List[int] = [2, 4, 8, 12, 16]

    def __init__(
            self,
            bert_l_param: int = None,
            bert_h_param: int = None
    ) -> None:
        #: The `bert_l_param` attribute represents the number of
        #: stacked encoders used for the BERT model. This
        #: attribute is set via the `BERT_L_PARAM`
        if bert_l_param is None:
            self.bert_l_param: int = int(
                os.getenv(
                    "BERT_L_PARAM",
                    "2"
                )
            )
        else:
            self.bert_l_param = bert_l_param

        if self.bert_l_param not in self.ALLOWED_L_VALUES:
            raise ValueError(
                "L value not allowed: "
                + "L param must an integer value of 2, 4, "
                + "6, 8, 10, 12, or 24"
            )

        if bert_h_param is None:
            self.bert_h_param: int = int(
                os.getenv(
                    "BERT_H_PARAM",
                    "128"
                )
            )
        else:
            self.bert_h_param = bert_h_param

        if self.bert_h_param == self.ALLOWED_H_VALUES[0]:
            self.bert_a_param = 2
        elif self.bert_h_param == self.ALLOWED_H_VALUES[1]:
            self.bert_a_param = 4
        elif self.bert_h_param == self.ALLOWED_H_VALUES[2]:
            self.bert_a_param = 8
        elif self.bert_h_param == self.ALLOWED_H_VALUES[3]:
            self.bert_a_param = 12
        elif self.bert_h_param == self.ALLOWED_H_VALUES[4]:
            self.bert_a_param = 16
        else:
            raise ValueError(
                "H value not allowed: "
                + "H param must an integer value of 128, 256, "
                + "512, 768, or 1024"
            )
