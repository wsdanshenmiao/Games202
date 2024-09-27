#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 100
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10
// ShadowMap的大小
#define SHADOW_MAP_SIZE 2048.0
// 光源大小
#define LIGHT_SIZE 0.04
// 近平面到视点的距离
#define NEAR_DEPTH 0.01
#define LIGHT_SIZE_MAP LIGHT_SIZE

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  // 获取打包后的深度
  vec4 rgbaDepth = texture2D(shadowMap, shadowCoord.xy);
  // 解包深度值
  float depth = unpack(rgbaDepth);
  float currentDepth = shadowCoord.z;
  // 比较深度值
  return depth + EPS > currentDepth ? 1.0 : 0.0;  
}

float PCF(sampler2D shadowMap, vec4 coords, float filterSize) {
  float sumV = 0.0;
  // 对ShadowMap进行采样
  for(int i = 0; i < PCF_NUM_SAMPLES; ++i){
    // 采样的纹理坐标
    vec4 shadowCoord = coords;
    shadowCoord.xy += filterSize * poissonDisk[i];
    sumV += useShadowMap(shadowMap, shadowCoord);
  }
  // 获取平均可见度
  float avgV = sumV / float(PCF_NUM_SAMPLES);

  return avgV;
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  int blockNum = 0;
  float sumDepth = 0.0;

  float filterSize = NEAR_DEPTH * float(LIGHT_SIZE_MAP) / zReceiver;

  for(int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i){
    float depth = unpack(texture2D(shadowMap, uv + filterSize * poissonDisk[i]));
    // 若物体的深度小于像素深度，遮挡
    if(depth - EPS < zReceiver){
      ++blockNum;
      sumDepth += depth;
    }
  }

  // 可能一个遮挡物都没有
	return blockNum == 0 ? -1.0 : sumDepth / float(blockNum);
}

float PCSS(sampler2D shadowMap, vec4 coords, out float filterSize, out float avgDepth){
  // STEP 1: avgblocker depth
  avgDepth = findBlocker(shadowMap, coords.xy, coords.z);
  //无遮挡物
  if(avgDepth == -1.0)
    return 1.0;
  // STEP 2: penumbra size
  filterSize = LIGHT_SIZE_MAP * (coords.z - avgDepth) / avgDepth;
  // STEP 3: filtering
  return PCF(shadowMap, coords, filterSize);
}


vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {
  
  float visibility;

  // 转化为NDC坐标
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  // 映射到[0,1]
  shadowCoord = (shadowCoord + vec3(1,1,1)) * 0.5;

  // 生成随机采样点
  poissonDiskSamples(shadowCoord.xy);
  //uniformDiskSamples(shadowCoord.xy);

  float filterSize, avgDepth;
  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0), 6.0 / float(SHADOW_MAP_SIZE));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0), filterSize, avgDepth);

  vec3 phongColor = blinnPhong();
  
  #define DEPTH 0
  #if DEPTH
  if(avgDepth == -1.0){
    gl_FragColor = vec4(shadowCoord.z, shadowCoord.z, shadowCoord.z, 1);
  }
  else{
    if(filterSize < 0.005){
      gl_FragColor = vec4(LIGHT_SIZE_MAP,shadowCoord.z,avgDepth,1);
    }
    else gl_FragColor = vec4(avgDepth, avgDepth, avgDepth, 1.0);
  }
  #else
  gl_FragColor = vec4(phongColor * visibility, 1);
  #endif
}