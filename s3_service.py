# s3_service.py - AWS S3 integration for face recognition images
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
import uuid
import logging
from typing import Optional, Tuple
from fastapi import UploadFile
import io
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        """Initialize S3 service with AWS credentials"""
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.bucket_prefix = os.getenv("AWS_S3_PREFIX", "lms-face-recognition")
        
        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.bucket_name]):
            raise ValueError("Missing required AWS S3 configuration. Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET_NAME")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ Connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            raise ValueError("AWS credentials not found")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise ValueError(f"S3 bucket '{self.bucket_name}' not found")
            else:
                raise ValueError(f"Error connecting to S3: {str(e)}")
    
    def upload_face_image(self, file: UploadFile, user_id: int, image_type: str) -> Tuple[str, str]:
        """
        Upload face image to S3 and return the S3 key and URL
        
        Args:
            file: UploadFile object
            user_id: User ID for organizing files
            image_type: Type of image (registration, verification, etc.)
            
        Returns:
            Tuple of (s3_key, s3_url)
        """
        try:
            # Generate unique filename
            file_extension = self._get_file_extension(file.filename)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{user_id}_{image_type}_{timestamp}_{unique_id}.{file_extension}"
            
            # Create S3 key with organized structure
            s3_key = f"{self.bucket_prefix}/users/{user_id}/{image_type}/{filename}"
            
            # Reset file pointer
            file.file.seek(0)
            file_content = file.file.read()
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                io.BytesIO(file_content),
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type or 'image/jpeg',
                    'Metadata': {
                        'user_id': str(user_id),
                        'image_type': image_type,
                        'upload_timestamp': datetime.utcnow().isoformat(),
                        'original_filename': file.filename or 'unknown'
                    }
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Uploaded {image_type} image for user {user_id} to S3: {s3_key}")
            return s3_key, s3_url
            
        except Exception as e:
            logger.error(f"❌ Error uploading image to S3: {str(e)}")
            raise Exception(f"Failed to upload image to S3: {str(e)}")
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for downloading an image
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL string
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            logger.error(f"❌ Error generating presigned URL for {s3_key}: {str(e)}")
            raise Exception(f"Failed to generate presigned URL: {str(e)}")
    
    def download_image(self, s3_key: str) -> bytes:
        """
        Download image from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Image bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"❌ Error downloading image from S3: {str(e)}")
            raise Exception(f"Failed to download image from S3: {str(e)}")
    
    def delete_image(self, s3_key: str) -> bool:
        """
        Delete image from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"🗑️ Deleted image from S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error deleting image from S3: {str(e)}")
            return False
    
    def list_user_images(self, user_id: int, image_type: Optional[str] = None) -> list:
        """
        List all images for a user
        
        Args:
            user_id: User ID
            image_type: Optional filter by image type
            
        Returns:
            List of image metadata
        """
        try:
            prefix = f"{self.bucket_prefix}/users/{user_id}/"
            if image_type:
                prefix += f"{image_type}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            images = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Get object metadata
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    images.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'metadata': head_response.get('Metadata', {}),
                        'url': f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{obj['Key']}"
                    })
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Error listing images for user {user_id}: {str(e)}")
            return []
    
    def cleanup_old_images(self, days_old: int = 30) -> int:
        """
        Clean up old verification images (keep registration images)
        
        Args:
            days_old: Delete images older than this many days
            
        Returns:
            Number of images deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # List verification images
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.bucket_prefix}/users/"
            )
            
            deleted_count = 0
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Only delete verification images, not registration images
                    if '/verification/' in obj['Key'] and obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.s3_client.delete_object(Bucket=self.bucket_name, Key=obj['Key'])
                        deleted_count += 1
                        
            logger.info(f"🧹 Cleaned up {deleted_count} old verification images")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")
            return 0
    
    def check_bucket_health(self) -> dict:
        """
        Check S3 bucket health and return stats
        
        Returns:
            Dictionary with bucket statistics
        """
        try:
            # Get bucket location
            location = self.s3_client.get_bucket_location(Bucket=self.bucket_name)
            
            # Count objects
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.bucket_prefix
            )
            
            total_objects = response.get('KeyCount', 0)
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            
            return {
                'status': 'healthy',
                'bucket_name': self.bucket_name,
                'region': location.get('LocationConstraint') or 'us-east-1',
                'total_objects': total_objects,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'prefix': self.bucket_prefix
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_file_extension(self, filename: Optional[str]) -> str:
        """Get file extension or default to jpg"""
        if filename and '.' in filename:
            return filename.split('.')[-1].lower()
        return 'jpg'