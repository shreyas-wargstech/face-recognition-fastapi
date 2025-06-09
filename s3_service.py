# s3_service.py - Updated with proper folder structure for scalability

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
import logging
from typing import Tuple, Dict, Optional
from fastapi import UploadFile
from datetime import datetime, timedelta
import hashlib
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        """Initialize S3 service with proper error handling"""
        try:
            # AWS credentials and configuration
            self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            self.bucket_name = os.getenv("S3_BUCKET_NAME")
            
            if not all([self.aws_access_key_id, self.aws_secret_access_key, self.bucket_name]):
                raise ValueError("Missing required AWS S3 configuration")
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Configuration for folder structure
            self.base_folder = os.getenv("S3_BASE_FOLDER", "lms-face-recognition")
            self.users_per_partition = int(os.getenv("USERS_PER_PARTITION", "1000"))  # 1000 users per partition
            
            logger.info(f"✅ S3 service initialized successfully - Bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 service: {str(e)}")
            raise
    
    def _get_user_partition(self, user_id: int) -> str:
        """
        Generate partition path for user based on ID ranges
        
        Examples:
        - User 1-1000: partition_0001
        - User 1001-2000: partition_0002
        - User 500001-501000: partition_0501
        """
        partition_number = ((user_id - 1) // self.users_per_partition) + 1
        return f"partition_{partition_number:04d}"
    
    def _get_date_partition(self) -> str:
        """
        Generate date-based partition for verification images
        Format: YYYY/MM to organize by year and month
        """
        now = datetime.utcnow()
        return f"{now.year}/{now.month:02d}"
    
    def _generate_s3_key(self, user_id: int, image_type: str, file_extension: str = "jpg") -> str:
        """
        Generate S3 key with proper folder structure
        
        Structure:
        - Registration: {base_folder}/registrations/{partition}/user_{user_id}/face_registration_{timestamp}.{ext}
        - Verification: {base_folder}/verifications/{date_partition}/{partition}/user_{user_id}/verification_{timestamp}_{hash}.{ext}
        
        Args:
            user_id: User ID
            image_type: 'registration', 'verification', 'verification_failed'
            file_extension: File extension
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        user_partition = self._get_user_partition(user_id)
        
        if image_type == "registration":
            # Registration images: organized by user partitions
            s3_key = f"{self.base_folder}/registrations/{user_partition}/user_{user_id}/face_registration_{timestamp}.{file_extension}"
        
        elif image_type in ["verification", "verification_failed"]:
            # Verification images: organized by date and user partitions for better lifecycle management
            date_partition = self._get_date_partition()
            
            # Add unique hash to prevent conflicts in high-frequency scenarios
            unique_hash = hashlib.md5(f"{user_id}_{timestamp}_{os.urandom(8).hex()}".encode()).hexdigest()[:8]
            
            status_folder = "failed" if image_type == "verification_failed" else "successful"
            s3_key = f"{self.base_folder}/verifications/{date_partition}/{status_folder}/{user_partition}/user_{user_id}/verification_{timestamp}_{unique_hash}.{file_extension}"
        
        else:
            # Fallback for any other image types
            s3_key = f"{self.base_folder}/misc/{user_partition}/user_{user_id}/{image_type}_{timestamp}.{file_extension}"
        
        return s3_key
    
    def upload_face_image(self, file: UploadFile, user_id: int, image_type: str) -> Tuple[str, str]:
        """
        Upload face image to S3 with proper folder structure
        
        Args:
            file: FastAPI UploadFile
            user_id: User ID
            image_type: Type of image ('registration', 'verification', 'verification_failed')
            
        Returns:
            Tuple of (s3_key, s3_url)
        """
        try:
            # Get file extension
            file_extension = "jpg"  # default
            if file.filename and "." in file.filename:
                file_extension = file.filename.split(".")[-1].lower()
            
            # Generate S3 key with proper folder structure
            s3_key = self._generate_s3_key(user_id, image_type, file_extension)
            
            # Reset file pointer
            file.file.seek(0)
            
            # Prepare metadata
            metadata = {
                'user_id': str(user_id),
                'image_type': image_type,
                'upload_timestamp': datetime.utcnow().isoformat(),
                'original_filename': file.filename or f"face_{image_type}.{file_extension}",
                'content_type': file.content_type or f"image/{file_extension}",
                'partition': self._get_user_partition(user_id)
            }
            
            # Add date partition for verification images
            if image_type in ["verification", "verification_failed"]:
                metadata['date_partition'] = self._get_date_partition()
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': file.content_type or f"image/{file_extension}",
                    'Metadata': metadata,
                    'ServerSideEncryption': 'AES256',  # Server-side encryption
                    'StorageClass': self._get_storage_class(image_type)
                }
            )
            
            # Generate S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            
            logger.info(f"📁 Uploaded {image_type} image for user {user_id} to {s3_key}")
            
            return s3_key, s3_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload image to S3: {str(e)}")
            raise Exception(f"S3 upload failed: {str(e)}")
    
    def _get_storage_class(self, image_type: str) -> str:
        """
        Determine appropriate S3 storage class based on image type
        
        - Registration images: STANDARD (frequently accessed)
        - Verification images: STANDARD_IA (infrequently accessed after initial period)
        """
        if image_type == "registration":
            return "STANDARD"  # Registration images may be accessed frequently
        elif image_type in ["verification", "verification_failed"]:
            return "STANDARD_IA"  # Verification images accessed less frequently
        else:
            return "STANDARD"
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL for secure access to S3 object
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL
        """
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return presigned_url
        except Exception as e:
            logger.error(f"❌ Failed to generate presigned URL for {s3_key}: {str(e)}")
            raise Exception(f"Failed to generate presigned URL: {str(e)}")
    
    def delete_image(self, s3_key: str) -> bool:
        """
        Delete image from S3
        
        Args:
            s3_key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"🗑️ Deleted image: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete image {s3_key}: {str(e)}")
            return False
    
    def cleanup_old_images(self, days: int = 30) -> int:
        """
        Clean up old verification images older than specified days
        Uses S3 lifecycle policies for efficient cleanup
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of deleted objects
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = 0
            
            # List objects in verification folders
            verification_prefixes = [
                f"{self.base_folder}/verifications/",
            ]
            
            for prefix in verification_prefixes:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                objects_to_delete = []
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                                objects_to_delete.append({'Key': obj['Key']})
                                
                                # Delete in batches of 1000 (S3 limit)
                                if len(objects_to_delete) >= 1000:
                                    response = self.s3_client.delete_objects(
                                        Bucket=self.bucket_name,
                                        Delete={'Objects': objects_to_delete}
                                    )
                                    deleted_count += len(response.get('Deleted', []))
                                    objects_to_delete = []
                
                # Delete remaining objects
                if objects_to_delete:
                    response = self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
                    deleted_count += len(response.get('Deleted', []))
            
            logger.info(f"🧹 Cleaned up {deleted_count} old verification images")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old images: {str(e)}")
            raise Exception(f"Cleanup failed: {str(e)}")
    
    def setup_lifecycle_policies(self):
        """
        Set up S3 lifecycle policies for automatic cleanup and cost optimization
        Call this during initial setup
        """
        try:
            lifecycle_configuration = {
                'Rules': [
                    {
                        'ID': 'VerificationImagesCleanup',
                        'Status': 'Enabled',
                        'Filter': {
                            'Prefix': f'{self.base_folder}/verifications/'
                        },
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            },
                            {
                                'Days': 365,
                                'StorageClass': 'DEEP_ARCHIVE'
                            }
                        ],
                        'Expiration': {
                            'Days': 2555  # 7 years retention for compliance
                        }
                    },
                    {
                        'ID': 'RegistrationImagesTransition',
                        'Status': 'Enabled',
                        'Filter': {
                            'Prefix': f'{self.base_folder}/registrations/'
                        },
                        'Transitions': [
                            {
                                'Days': 90,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 365,
                                'StorageClass': 'GLACIER'
                            }
                        ]
                        # No expiration for registration images
                    },
                    {
                        'ID': 'FailedVerificationCleanup',
                        'Status': 'Enabled',
                        'Filter': {
                            'Prefix': f'{self.base_folder}/verifications/'
                        },
                        'AbortIncompleteMultipartUpload': {
                            'DaysAfterInitiation': 7
                        }
                    }
                ]
            }
            
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_configuration
            )
            
            logger.info("✅ S3 lifecycle policies configured successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup lifecycle policies: {str(e)}")
            raise Exception(f"Lifecycle policy setup failed: {str(e)}")
    
    def check_bucket_health(self) -> Dict:
        """
        Check S3 bucket health and get statistics
        
        Returns:
            Dictionary with bucket health information
        """
        try:
            # Check bucket access
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            
            # Get bucket statistics
            total_objects = 0
            total_size = 0
            partition_stats = {}
            
            # Sample a few partitions to get statistics
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name, 
                Prefix=f"{self.base_folder}/",
                MaxKeys=1000  # Limit for health check
            )
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_objects += 1
                        total_size += obj['Size']
                        
                        # Extract partition info from key
                        key_parts = obj['Key'].split('/')
                        if len(key_parts) >= 4 and 'partition_' in key_parts[2]:
                            partition = key_parts[2]
                            if partition not in partition_stats:
                                partition_stats[partition] = {'objects': 0, 'size': 0}
                            partition_stats[partition]['objects'] += 1
                            partition_stats[partition]['size'] += obj['Size']
            
            return {
                "status": "healthy",
                "bucket_name": self.bucket_name,
                "region": self.aws_region,
                "total_objects_sampled": total_objects,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "partition_stats": dict(list(partition_stats.items())[:10]),  # Show first 10 partitions
                "users_per_partition": self.users_per_partition,
                "base_folder": self.base_folder,
                "storage_classes": ["STANDARD", "STANDARD_IA", "GLACIER", "DEEP_ARCHIVE"],
                "lifecycle_policies": "configured",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except NoCredentialsError:
            return {
                "status": "error",
                "error": "AWS credentials not found or invalid",
                "timestamp": datetime.utcnow().isoformat()
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            return {
                "status": "error",
                "error": f"AWS Error: {error_code} - {e.response['Error']['Message']}",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_user_images(self, user_id: int, image_type: Optional[str] = None) -> list[Dict]:
        """
        Get all images for a specific user
        
        Args:
            user_id: User ID
            image_type: Optional filter by image type
            
        Returns:
            List of image metadata
        """
        try:
            user_partition = self._get_user_partition(user_id)
            images = []
            
            # Define search prefixes based on image type
            if image_type == "registration":
                prefixes = [f"{self.base_folder}/registrations/{user_partition}/user_{user_id}/"]
            elif image_type in ["verification", "verification_failed"]:
                prefixes = [f"{self.base_folder}/verifications/"]
            else:
                # Search all folders
                prefixes = [
                    f"{self.base_folder}/registrations/{user_partition}/user_{user_id}/",
                    f"{self.base_folder}/verifications/"
                ]
            
            for prefix in prefixes:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            # Filter by user_id in verification paths
                            if 'verifications/' in obj['Key'] and f"user_{user_id}/" not in obj['Key']:
                                continue
                            
                            # Get object metadata
                            try:
                                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=obj['Key'])
                                metadata = response.get('Metadata', {})
                                
                                images.append({
                                    'key': obj['Key'],
                                    'size': obj['Size'],
                                    'last_modified': obj['LastModified'].isoformat(),
                                    'storage_class': obj.get('StorageClass', 'STANDARD'),
                                    'image_type': metadata.get('image_type', 'unknown'),
                                    'partition': metadata.get('partition', user_partition),
                                    'original_filename': metadata.get('original_filename'),
                                    'presigned_url': self.get_presigned_url(obj['Key'], 3600)
                                })
                            except Exception as e:
                                logger.warning(f"Failed to get metadata for {obj['Key']}: {str(e)}")
            
            return images
            
        except Exception as e:
            logger.error(f"❌ Failed to get user images for user {user_id}: {str(e)}")
            raise Exception(f"Failed to get user images: {str(e)}")
    
    def migrate_existing_images(self, batch_size: int = 100):
        """
        Migrate existing images to new folder structure
        Use this for migrating from old flat structure to new partitioned structure
        """
        try:
            migrated_count = 0
            
            # List all objects in old structure
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=f"{self.base_folder}/")
            
            for page in pages:
                if 'Contents' in page:
                    batch_operations = []
                    
                    for obj in page['Contents']:
                        old_key = obj['Key']
                        
                        # Skip if already in new structure
                        if '/partition_' in old_key or '/registrations/' in old_key or '/verifications/' in old_key:
                            continue
                        
                        # Extract user_id from old key pattern
                        try:
                            # Assume old pattern like: lms-face-recognition/user_123_registration_20240101.jpg
                            filename = old_key.split('/')[-1]
                            if 'user_' in filename:
                                user_id = int(filename.split('user_')[1].split('_')[0])
                                
                                # Determine image type from filename
                                if 'registration' in filename:
                                    image_type = 'registration'
                                elif 'verification' in filename:
                                    image_type = 'verification'
                                else:
                                    image_type = 'misc'
                                
                                # Generate new key
                                file_extension = filename.split('.')[-1] if '.' in filename else 'jpg'
                                new_key = self._generate_s3_key(user_id, image_type, file_extension)
                                
                                # Add to batch operations
                                batch_operations.append({
                                    'old_key': old_key,
                                    'new_key': new_key,
                                    'user_id': user_id,
                                    'image_type': image_type
                                })
                                
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse user_id from {old_key}: {str(e)}")
                            continue
                    
                    # Process batch operations
                    for operation in batch_operations[:batch_size]:
                        try:
                            # Copy to new location
                            self.s3_client.copy_object(
                                CopySource={'Bucket': self.bucket_name, 'Key': operation['old_key']},
                                Bucket=self.bucket_name,
                                Key=operation['new_key'],
                                MetadataDirective='REPLACE',
                                Metadata={
                                    'user_id': str(operation['user_id']),
                                    'image_type': operation['image_type'],
                                    'migration_timestamp': datetime.utcnow().isoformat(),
                                    'partition': self._get_user_partition(operation['user_id'])
                                }
                            )
                            
                            # Delete old object
                            self.s3_client.delete_object(Bucket=self.bucket_name, Key=operation['old_key'])
                            
                            migrated_count += 1
                            logger.info(f"Migrated: {operation['old_key']} -> {operation['new_key']}")
                            
                        except Exception as e:
                            logger.error(f"Failed to migrate {operation['old_key']}: {str(e)}")
            
            logger.info(f"✅ Migration completed. Migrated {migrated_count} images")
            return migrated_count
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {str(e)}")
            raise Exception(f"Migration failed: {str(e)}")