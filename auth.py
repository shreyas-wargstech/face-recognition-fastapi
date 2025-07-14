# auth.py - JWT Authentication for Face Recognition API

import jwt
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# JWT Configuration - should match your main backend
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security scheme
security = HTTPBearer()

class AuthenticatedUser:
    """User model extracted from JWT token - matches frontend UserType"""
    def __init__(self, token_payload: Dict[str, Any]):
        self.id: int = token_payload.get('id')
        self.salutation: str = token_payload.get('salutation', '')
        self.isPopupFlag: bool = token_payload.get('isPopupFlag', False)
        self.name: str = token_payload.get('name', '')
        self.mobile: str = token_payload.get('mobile', '')
        self.email: str = token_payload.get('email', '')
        self.specialization: str = token_payload.get('specialization', '')
        self.avatarID: int = token_payload.get('avatarID', 0)
        self.profilePictureUrl: str = token_payload.get('profilePictureUrl', '')
        self.role: str = token_payload.get('role', 'Student')
        self.privileges: List[str] = token_payload.get('privileges', [])
        self.iat: int = token_payload.get('iat', 0)
        self.exp: int = token_payload.get('exp', 0)
        self.qualifications: str = token_payload.get('qualifications', '')
        
        # Additional fields for face recognition context
        self.roleId: int = token_payload.get('roleId', 1)
        self.active: bool = token_payload.get('active', True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging"""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'privileges': self.privileges
        }
    
    def has_privilege(self, privilege: str) -> bool:
        """Check if user has specific privilege"""
        return privilege in self.privileges
    
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role.lower() in ['admin', 'administrator']
    
    def is_instructor(self) -> bool:
        """Check if user is instructor"""
        return self.role.lower() in ['instructor', 'teacher']
    
    def can_access_face_recognition(self) -> bool:
        """Check if user can access face recognition features"""
        # All authenticated users can access face recognition
        return self.active and self.id > 0

class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )

class PermissionError(HTTPException):
    """Custom permission error"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )

def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate JWT token - matches frontend jwt.decode logic
    """
    try:
        # Decode the token
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        # Check expiration
        current_time = datetime.utcnow().timestamp()
        if payload.get('exp', 0) < current_time:
            raise AuthenticationError("Token has expired")
        
        # Validate required fields
        if not payload.get('id'):
            raise AuthenticationError("Invalid token: missing user ID")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
    except Exception as e:
        logger.error(f"JWT decode error: {str(e)}")
        raise AuthenticationError("Token validation failed")

def create_jwt_token(user_data: Dict[str, Any]) -> str:
    """
    Create JWT token for user (for testing or token refresh)
    """
    try:
        # Set expiration
        exp_time = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        
        payload = {
            **user_data,
            'iat': datetime.utcnow().timestamp(),
            'exp': exp_time.timestamp()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
        
    except Exception as e:
        logger.error(f"JWT creation error: {str(e)}")
        raise Exception("Failed to create token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> AuthenticatedUser:
    """
    Extract and validate user from JWT token - main auth dependency
    Matches frontend token handling logic
    """
    try:
        # Extract token from Authorization header
        token = credentials.credentials
        
        if not token:
            raise AuthenticationError("No token provided")
        
        # Decode and validate token
        payload = decode_jwt_token(token)
        
        # Create authenticated user object
        user = AuthenticatedUser(payload)
        
        # Additional validation
        if not user.can_access_face_recognition():
            raise AuthenticationError("User not authorized for face recognition")
        
        logger.info(f"Authenticated user: {user.name} (ID: {user.id})")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise AuthenticationError("Authentication failed")

async def get_current_active_user(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """
    Get current user and ensure they are active
    """
    if not current_user.active:
        raise AuthenticationError("User account is inactive")
    
    return current_user

async def get_admin_user(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """
    Require admin privileges
    """
    if not current_user.is_admin():
        raise PermissionError("Admin privileges required")
    
    return current_user

async def get_instructor_or_admin_user(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """
    Require instructor or admin privileges
    """
    if not (current_user.is_instructor() or current_user.is_admin()):
        raise PermissionError("Instructor or admin privileges required")
    
    return current_user

def require_privilege(privilege: str):
    """
    Decorator for requiring specific privilege
    """
    async def privilege_checker(
        current_user: AuthenticatedUser = Depends(get_current_user)
    ) -> AuthenticatedUser:
        if not current_user.has_privilege(privilege):
            raise PermissionError(f"Required privilege: {privilege}")
        return current_user
    
    return privilege_checker

def validate_user_access(user_id: int, current_user: AuthenticatedUser) -> bool:
    """
    Check if current user can access data for specified user_id
    - Users can access their own data
    - Admins can access anyone's data
    - Instructors can access student data (if implemented)
    """
    if current_user.id == user_id:
        return True
    
    if current_user.is_admin():
        return True
    
    # Additional logic for instructors accessing student data can be added here
    # if current_user.is_instructor():
    #     return check_instructor_student_relationship(current_user.id, user_id)
    
    return False

async def verify_user_access(
    user_id: int,
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """
    Dependency to verify user can access specified user_id's data
    """
    if not validate_user_access(user_id, current_user):
        raise PermissionError("Access denied: cannot access this user's data")
    
    return current_user

# Optional: Create a test token for development
def create_test_token(user_id: int = 1, name: str = "Test User", role: str = "Student") -> str:
    """
    Create a test token for development/testing
    """
    test_user_data = {
        'id': user_id,
        'name': name,
        'email': f'test{user_id}@example.com',
        'mobile': '1234567890',
        'role': role,
        'privileges': ['face_recognition'],
        'active': True,
        'roleId': 1 if role == 'Student' else 2,
        'salutation': 'Mr.',
        'isPopupFlag': False,
        'specialization': 'General',
        'avatarID': 1,
        'profilePictureUrl': '',
        'qualifications': ''
    }
    
    return create_jwt_token(test_user_data)

# Health check for auth system
async def auth_health_check() -> Dict[str, Any]:
    """
    Check authentication system health
    """
    try:
        # Test token creation and validation
        test_token = create_test_token(999, "Health Check User")
        test_payload = decode_jwt_token(test_token)
        
        return {
            "status": "healthy",
            "jwt_algorithm": JWT_ALGORITHM,
            "token_expiration_hours": JWT_EXPIRATION_HOURS,
            "test_successful": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
