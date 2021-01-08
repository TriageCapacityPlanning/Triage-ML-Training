from .models.radius_variance import RadiusVariance


def train():
    rv_model = RadiusVariance()
    print(rv_model.get_model().summary())
    pass
