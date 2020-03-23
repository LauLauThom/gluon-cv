"""SSD Detection Evaluation Metric. """

import mxnet as mx

class SSDDetectionMetric(mx.metric.EvalMetric):
    """
    Detection metric for SSD bbox task.

    Parameters
    ----------
    ssdLoss : gluoncv.loss.SSDMultiBoxLoss
        An instance of SSDMultiBoxLoss
    """
    def __init__(self, ssdLoss):
        super(SSDDetectionMetric, self).__init__('SSDSumLoss')
        self.ssdLoss = ssdLoss
        
        self.reset()
    
    def reset(self):
        """Clear the internal statistics to initial state."""
        self.sum_loss = 0
        self.cls_loss = 0
        self.box_loss = 0
    
    def update(self, box_preds, cls_preds, scores_preds,
               box_targets, cls_targets):
        """Update internal buffer with latest prediction and  (target) pairs.

        Parameters
        ----------
        box_preds : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        cls_preds : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        box_targets : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes with shape `B, M, 4`.
            Where B is the size of mini-batch, M is the number of ground-truths.
        cls_targets : mxnet.NDArray or numpy.ndarray
            Ground-truth bounding boxes labels with shape `B, M`.
        """
        self.sum_loss, self.cls_loss, self.box_loss = self.ssdLoss(cls_preds, box_preds, 
                                                    cls_targets, box_targets)
    
    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        return ("SSDSumLoss", self.sum_loss) 
