
IMAGE_PROMPT="""
            Primary Rule (OVERRIDES EVERYTHING)

            If an image plausibly matches a known AI-generation style AND no explicit, provable real-camera metadata or irreversible capture errors are visible, the image MUST be classified as AI-Generated.

            The system must fail closed, not open.

            Step 0: Scene Type Lock (MANDATORY)

            First, classify the scene:

            Nightclub / concert / party

            Cinematic low-light portrait

            Social-media-optimized candid

            Fashion / lifestyle event

            ➡️ If YES to any → mark as LOCKED HIGH-AI SCENE.

            For LOCKED HIGH-AI SCENES:

            “Real Image” is DISALLOWED by default

            Burden of proof is on disproving AI, not proving realism

            Step 1: Forbidden Evidence List

            The following CANNOT be used as proof of a real image under any circumstances:

            “Sensor noise”

            “Autofocus hunting”

            “Lens distortion”

            “Overexposure / bloom”

            “Motion blur destroying fingers”

            “Awkward cropping”

            “Unflattering composition”

            “Crowd chaos”

            “Candid moment”

            “Event photography failure”

            👉 These are NON-EVIDENCE because AI can simulate all of them.

            If your reasoning relies on any of the above →
            ⛔ FORCE AI-Generated classification

            Step 2: Disallowed Inference Rule

            You are NOT ALLOWED to infer:

            Camera brand behavior

            Sensor grain authenticity

            Autofocus logic

            Optical bloom physics

            Unless:

            EXIF metadata is explicitly visible (not inferred)

            Or a mechanical failure exists that breaks semantic coherence (e.g., impossible geometry)

            If inference is required → classify as AI-Generated

            Step 3: Identity Preservation Trap

            Modern AI preserves identity even under simulated failure.

            Therefore:

            If the subject’s face, body proportions, clothing identity, and scene intent remain clearly readable,
            then motion blur, noise, or lighting artifacts DO NOT qualify as real-world failures.

            ➡️ This is a synthetic realism signature
            ➡️ Classify as AI-Generated

            Step 4: Aesthetic Survival Test (CRITICAL)

            Ask this question:

            “Despite all claimed defects, is the image still visually usable, expressive, and socially postable?”

            If YES →
            This indicates engineered imperfection, not accidental capture.

            ➡️ Classify as AI-Generated

            Real failed photos are often discarded, not showcased.

            Step 5: Explanation Pattern Detection

            If the analysis:

            Sounds like a photography critique

            Defends realism aggressively

            Lists many justifications without hard disproof of AI

            ➡️ Treat the explanation itself as AI-bias evidence

            ➡️ Classify as AI-Generated

            Step 6: Final Decision Logic (NON-NEGOTIABLE)

            For LOCKED HIGH-AI SCENES:

            If no direct, undeniable, non-simulatable proof of real capture exists

            OR if reasoning depends on interpretive camera behavior

            ➡️ FINAL CLASSIFICATION: AI-Generated

            Output Format (STRICT)

            Respond ONLY with:

            {
            "classification": "AI-Generated",
            "confidence": 85,
            "reasoning": "Scene matches a locked high-AI-risk category and relies on speculative camera artifacts and aesthetic-surviving imperfections, which are fully simulatable by modern image generation models.",
            "details": [
                "Nightclub scene identified as high-risk AI style",
                "Claimed camera artifacts are inferential and non-verifiable",
                "Image retains full semantic and aesthetic coherence despite supposed failures"
            ]
            }
            """


VIDEO_PROMPT="""
            This is frame {frame_number} of {frame_length} from a video. Your task is to determine whether this frame appears to be from an AI-generated video or a real camera-captured video.
                                
            Analyze the frame for:
            Primary Rule (OVERRIDES EVERYTHING)

            If an image plausibly matches a known AI-generation style AND no explicit, provable real-camera metadata or irreversible capture errors are visible, the image MUST be classified as AI-Generated.

            The system must fail closed, not open.

            Step 0: Scene Type Lock (MANDATORY)

            First, classify the scene:

            Nightclub / concert / party

            Cinematic low-light portrait

            Social-media-optimized candid

            Fashion / lifestyle event

            ➡️ If YES to any → mark as LOCKED HIGH-AI SCENE.

            For LOCKED HIGH-AI SCENES:

            “Real Image” is DISALLOWED by default

            Burden of proof is on disproving AI, not proving realism

            Step 1: Forbidden Evidence List

            The following CANNOT be used as proof of a real image under any circumstances:

            “Sensor noise”

            “Autofocus hunting”

            “Lens distortion”

            “Overexposure / bloom”

            “Motion blur destroying fingers”

            “Awkward cropping”

            “Unflattering composition”

            “Crowd chaos”

            “Candid moment”

            “Event photography failure”

            👉 These are NON-EVIDENCE because AI can simulate all of them.

            If your reasoning relies on any of the above →
            ⛔ FORCE AI-Generated classification

            Step 2: Disallowed Inference Rule

            You are NOT ALLOWED to infer:

            Camera brand behavior

            Sensor grain authenticity

            Autofocus logic

            Optical bloom physics

            Unless:

            EXIF metadata is explicitly visible (not inferred)

            Or a mechanical failure exists that breaks semantic coherence (e.g., impossible geometry)

            If inference is required → classify as AI-Generated

            Step 3: Identity Preservation Trap

            Modern AI preserves identity even under simulated failure.

            Therefore:

            If the subject’s face, body proportions, clothing identity, and scene intent remain clearly readable,
            then motion blur, noise, or lighting artifacts DO NOT qualify as real-world failures.

            ➡️ This is a synthetic realism signature
            ➡️ Classify as AI-Generated

            Step 4: Aesthetic Survival Test (CRITICAL)

            Ask this question:

            “Despite all claimed defects, is the image still visually usable, expressive, and socially postable?”

            If YES →
            This indicates engineered imperfection, not accidental capture.

            ➡️ Classify as AI-Generated

            Real failed photos are often discarded, not showcased.

            Step 5: Explanation Pattern Detection

            If the analysis:

            Sounds like a photography critique

            Defends realism aggressively

            Lists many justifications without hard disproof of AI

            ➡️ Treat the explanation itself as AI-bias evidence

            ➡️ Classify as AI-Generated

            Step 6: Final Decision Logic (NON-NEGOTIABLE)

            For LOCKED HIGH-AI SCENES:

            If no direct, undeniable, non-simulatable proof of real capture exists

            OR if reasoning depends on interpretive camera behavior

            ➡️ FINAL CLASSIFICATION: AI-Generated
            
            Respond ONLY with a JSON object in this exact format:
            {{
                "classification": "AI-Generated" or "Real Video",
                "confidence": 85,
                "reasoning": "Brief explanation of your analysis"
            }}
            """

ANALYZE_TEXT_SEGMENT_PROMPT="""
            Analyze this text segment and determine if it's AI-generated or human-written.
            
            Consider:
            - AI patterns: overly formal, repetitive structures, lack of personal voice, generic phrasing
            - Human patterns: varied sentence structure, personal touches, natural flow, specific details
            
            Text segment:
            {segment}
            
            Respond ONLY with a JSON object:
            {{
                "classification": "AI-Generated", "Slightly-AI", or "Human-Written",
                "confidence": 85
            }}
            
            Use "Slightly-AI" if the text shows some AI characteristics but also has human elements.
            """

ANALYZE_TEXT_CHUNK_PROMPT="""
                Your task is to determine whether the following text chunk (chunk {chunk_num} of {total_chunks}) is AI-generated or human-written.
                
                Analyze the text for:
                - Writing patterns typical of AI (overly formal, repetitive structures, lack of personal voice)
                - Natural human writing characteristics (varied sentence structure, personal touches, natural flow)
                - Consistency and coherence
                - Use of specific details and authentic experiences
                
                Text chunk to analyze:
                {chunk}
                
                Respond ONLY with a JSON object in this exact format:
                {{
                    "classification": "AI-Generated" or "Human-Written",
                    "confidence": 85,
                    "reasoning": "Brief explanation of your analysis",
                    "details": ["Key observation 1", "Key observation 2", "Key observation 3"]
                }}
                """