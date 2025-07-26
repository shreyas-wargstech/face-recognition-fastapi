from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import Optional
import os
from enum import Enum

security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "may_the_force_be_with_you")

class RequestType(str, Enum):
        registration = "registration"
        verification = "verification"

class UserAuthDTO(BaseModel):
    """User Authentication Data Transfer Object"""
    id: Optional[int] = None
    assessmentID: Optional[int] = None
    contentID: Optional[int] = None
    requestType: Optional[RequestType] = None

    @classmethod
    def from_token_payload(cls, payload: dict):
        """Create UserAuthDTO from JWT token payload - matches your JS fromTokenPayload"""
        if payload.get("requestType") == "verification":
            return cls(
                requestType=RequestType.verification,
                id=payload.get("userId"),
                assessmentID=payload.get("assessmentID"),
                contentID=payload.get("contentID")
            )
        elif payload.get("requestType") == "registration":
            return cls(
                requestType=RequestType.registration,
                id=payload.get("userId")
            )
        else:
            raise ValueError("Invalid request type in token payload")


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