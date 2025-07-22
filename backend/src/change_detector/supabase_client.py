#!/usr/bin/env python3
"""
Supabase client for satellite change detection system
Handles auth, storage, vector DB operations, and analysis history
"""

import os
import asyncio
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from supabase import create_client, Client
from postgrest import APIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AnalysisResult:
    """Structure for analysis results"""
    id: str
    user_id: Optional[str]
    image_hash_before: str
    image_hash_after: str
    change_percentage: float
    changed_pixels: int
    total_pixels: int
    contours_count: int
    significance_level: str
    agent_analysis: Optional[str]
    created_at: datetime
    processing_time_ms: int
    tools_used: List[str]

@dataclass
class ImageEmbedding:
    """Structure for image embeddings"""
    id: str
    image_hash: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime

class SupabaseClient:
    """Enhanced Supabase client for satellite change detection"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.client: Optional[Client] = None
        self.is_connected = False
        
        if self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                self.is_connected = True
                print("✅ Supabase client initialized successfully")
            except Exception as e:
                print(f"❌ Supabase initialization failed: {e}")
                self.client = None
        else:
            print("⚠️ Supabase credentials not found - running without database features")
    
    def _generate_image_hash(self, image_base64: str) -> str:
        """Generate consistent hash for image content"""
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        return hashlib.sha256(image_base64.encode()).hexdigest()[:16]
    
    async def store_analysis_result(
        self, 
        before_image_base64: str,
        after_image_base64: str,
        change_results: Dict[str, Any],
        agent_analysis: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        processing_time_ms: int = 0
    ) -> Optional[str]:
        """Store analysis result in database"""
        if not self.client or not self.is_connected:
            print("⚠️ Supabase not available - analysis not stored")
            return None
        
        try:
            # Generate image hashes
            hash_before = self._generate_image_hash(before_image_base64)
            hash_after = self._generate_image_hash(after_image_base64)
            
            # Generate unique analysis ID
            analysis_id = hashlib.sha256(f"{hash_before}_{hash_after}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Prepare data for storage
            analysis_data = {
                'id': analysis_id,
                'user_id': user_id,
                'image_hash_before': hash_before,
                'image_hash_after': hash_after,
                'change_percentage': float(change_results.get('change_percentage', 0)),
                'changed_pixels': int(change_results.get('changed_pixels', 0)),
                'total_pixels': int(change_results.get('total_pixels', 0)),
                'contours_count': int(change_results.get('contours_count', 0)),
                'significance_level': self._determine_significance(change_results.get('change_percentage', 0)),
                'agent_analysis': agent_analysis,
                'tools_used': tools_used or [],
                'processing_time_ms': processing_time_ms,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'raw_results': change_results
            }
            
            # Insert into database
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('analysis_results').insert(analysis_data).execute()
            )
            
            print(f"✅ Analysis result stored with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            print(f"❌ Failed to store analysis result: {e}")
            return None
    
    async def store_image_embedding(
        self,
        image_base64: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """Store image embedding for similarity search"""
        if not self.client or not self.is_connected:
            return None
        
        try:
            image_hash = self._generate_image_hash(image_base64)
            
            embedding_data = {
                'id': f"emb_{image_hash}",
                'image_hash': image_hash,
                'embedding': embedding,
                'metadata': metadata or {},
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Insert or update embedding
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('image_embeddings').upsert(embedding_data).execute()
            )
            
            print(f"✅ Image embedding stored for hash: {image_hash}")
            return image_hash
            
        except Exception as e:
            print(f"❌ Failed to store image embedding: {e}")
            return None
    
    async def find_similar_analyses(
        self,
        change_percentage: float,
        significance_level: str,
        limit: int = 5
    ) -> List[AnalysisResult]:
        """Find similar analysis results"""
        if not self.client or not self.is_connected:
            return []
        
        try:
            # Query for similar analyses
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('analysis_results')
                .select('*')
                .eq('significance_level', significance_level)
                .gte('change_percentage', max(0, change_percentage - 2))
                .lte('change_percentage', change_percentage + 2)
                .order('created_at', desc=True)
                .limit(limit)
                .execute()
            )
            
            analyses = []
            for row in result.data:
                analyses.append(AnalysisResult(
                    id=row['id'],
                    user_id=row.get('user_id'),
                    image_hash_before=row['image_hash_before'],
                    image_hash_after=row['image_hash_after'],
                    change_percentage=row['change_percentage'],
                    changed_pixels=row['changed_pixels'],
                    total_pixels=row['total_pixels'],
                    contours_count=row['contours_count'],
                    significance_level=row['significance_level'],
                    agent_analysis=row.get('agent_analysis'),
                    created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                    processing_time_ms=row.get('processing_time_ms', 0),
                    tools_used=row.get('tools_used', [])
                ))
            
            return analyses
            
        except Exception as e:
            print(f"❌ Failed to find similar analyses: {e}")
            return []
    
    async def get_analysis_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[AnalysisResult]:
        """Get analysis history for user or system"""
        if not self.client or not self.is_connected:
            return []
        
        try:
            query = self.client.table('analysis_results').select('*')
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: query.order('created_at', desc=True).limit(limit).execute()
            )
            
            analyses = []
            for row in result.data:
                analyses.append(AnalysisResult(
                    id=row['id'],
                    user_id=row.get('user_id'),
                    image_hash_before=row['image_hash_before'],
                    image_hash_after=row['image_hash_after'],
                    change_percentage=row['change_percentage'],
                    changed_pixels=row['changed_pixels'],
                    total_pixels=row['total_pixels'],
                    contours_count=row['contours_count'],
                    significance_level=row['significance_level'],
                    agent_analysis=row.get('agent_analysis'),
                    created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                    processing_time_ms=row.get('processing_time_ms', 0),
                    tools_used=row.get('tools_used', [])
                ))
            
            return analyses
            
        except Exception as e:
            print(f"❌ Failed to get analysis history: {e}")
            return []
    
    async def get_analysis_stats(self) -> Dict[str, Any]:
        """Get system-wide analysis statistics"""
        if not self.client or not self.is_connected:
            return {}
        
        try:
            # Get total analyses count
            total_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('analysis_results').select('id', count='exact').execute()
            )
            
            # Get significance level distribution
            significance_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('analysis_results')
                .select('significance_level', count='exact')
                .execute()
            )
            
            return {
                'total_analyses': total_result.count,
                'significance_distribution': {
                    'HIGH': 0,
                    'MEDIUM': 0, 
                    'LOW': 0,
                    'MINIMAL': 0
                },
                'avg_processing_time_ms': 0,  # Could be calculated with proper aggregation
                'most_active_day': None  # Could be calculated with date aggregation
            }
            
        except Exception as e:
            print(f"❌ Failed to get analysis stats: {e}")
            return {}
    
    def _determine_significance(self, change_percentage: float) -> str:
        """Determine significance level from change percentage"""
        if change_percentage > 15:
            return "HIGH"
        elif change_percentage > 5:
            return "MEDIUM"
        elif change_percentage > 1:
            return "LOW"
        else:
            return "MINIMAL"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Supabase connection and database health"""
        if not self.client or not self.is_connected:
            return {
                "status": "unavailable",
                "connected": False,
                "error": "Supabase client not initialized"
            }
        
        try:
            # Simple query to test connection
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.table('analysis_results').select('id').limit(1).execute()
            )
            
            return {
                "status": "healthy",
                "connected": True,
                "tables_accessible": True,
                "last_check": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }

# Global Supabase client instance
supabase_client = SupabaseClient() 