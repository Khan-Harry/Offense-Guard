from routes.auth_routes import auth_bp
from routes.detection_routes import detection_bp
from routes.override_routes import override_bp
from routes.admin_routes import admin_bp
from routes.history_routes import history_bp
from routes.download_routes import download_bp

__all__ = [
    'auth_bp',
    'detection_bp',
    'override_bp',
    'admin_bp',
    'history_bp',
    'download_bp'
]
