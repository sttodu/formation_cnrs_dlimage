class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, prediction, target, weight):
        assert prediction.size() == target.size(), "'prediction' and 'target' must have the same shape"
        prediction = flatten(prediction) #flatten all dimensions except channel/class
        target = flatten(target)
        target = target.float()

        if prediction.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            prediction = torch.cat((prediction, 1 - prediction), dim=0)
            target = torch.cat((target, 1 - target), dim=0)
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (prediction * target).sum(-1)
        print(intersect.shape)
        intersect = intersect * w_l

        denominator = (prediction + target).sum(-1)
        print(denominator)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 1 - (2 * (intersect.sum() / denominator.sum()))

GeneralizedDiceLoss(normalization='softmax').dice(prediction=out, target=seg, weight=None)
