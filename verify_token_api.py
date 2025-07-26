from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional, List
import os

security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "may_the_force_be_with_you")

class UserAuthDTO(BaseModel):
    """User Authentication Data Transfer Object"""
    id: Optional[int] = None
    isPopupFlag: bool = False
    avatarID: Optional[str] = None
    profilePictureUrl: Optional[str] = None
    salutation: Optional[str] = None
    name: Optional[str] = None
    mobile: Optional[str] = None
    qualifications: Optional[str] = None
    email: Optional[str] = None
    specialization: Optional[str] = None
    role: Optional[str] = None
    privileges: List[str] = []

    @classmethod
    def from_token_payload(cls, payload: dict):
        """Create UserAuthDTO from JWT token payload - matches your JS fromTokenPayload"""
        return cls(
            id=payload.get('id'),
            isPopupFlag=payload.get('isPopupFlag', False),
            avatarID=payload.get('AvatarID'),
            profilePictureUrl=payload.get('ProfilePictureUrl'),
            salutation=payload.get('salutation'),
            name=payload.get('name'),
            mobile=payload.get('mobile'),
            qualifications=payload.get('qualifications'),
            email=payload.get('email'),
            specialization=payload.get('specialization'),
            role=payload.get('role'),
            privileges=payload.get('privileges', [])
        )


def verify_jwt_token(credentials = Depends(security)) -> UserAuthDTO:
    """Verify JWT token from main backend"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token using the same secret as main backend
        decoded = jwt.decode(
            credentials.credentials, 
            JWT_SECRET,
            algorithms=["HS256"]
        )
        
        user_auth_dto = UserAuthDTO.from_token_payload(decoded)

        return user_auth_dto
        
    except JWTError as e:
        print(f"JWT Error: {e}")  # For debugging
        raise credentials_exception