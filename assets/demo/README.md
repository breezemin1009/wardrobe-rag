# 표시용 의류 사진

2026-10-14에 내장 imagegen 도구로 생성했다. 실제 상품, 실제 사용자의 소유 의류 또는 호환성 평가의 정답 이미지가 아니다. 등록된 사용자 사진을 덮어쓰지 않는다. 앱은 원본 시트의 각 칸을 표시할 때 잘라 보여 주며 원본 픽셀 파일은 보존한다.

| 파일 | 구성 |
|---|---|
| garments-studio-v2.png | 흰 셔츠, 크림 티셔츠, 블루 셔츠, 블랙 니트, 그레이 스웨트셔츠, 올리브 셔츠, 블랙 팬츠, 블루 데님, 베이지 팬츠 |
| garments-patterns-v2.png | 블루 스트라이프, 브레톤 티셔츠, 올리브 체크, 베이지 니트, 크림 니트, 네이비 폴로, 그레이 팬츠, 올리브 팬츠, 크림 진 |
| registration-blue-shirt.png | 사진 등록 및 NVIDIA API 연결 확인용 블루 셔츠 한 장 |

## 사용한 프롬프트

첫 시트: “Photorealistic studio flat-lay photography, real cotton weave, knit texture, subtle natural wrinkles, crisp seams, accurate solid colours, soft diffused daylight, consistent very pale warm grey background. Exactly 3 columns by 3 rows. Each cell contains one complete isolated garment, front view, centered with generous margin. Row 1: white Oxford button-down shirt, cream crew-neck cotton T-shirt, light blue Oxford shirt. Row 2: black knit sweater, grey sweatshirt, olive overshirt. Row 3: black tailored trousers, medium blue jeans, beige chinos. No text, numbers, logos, props, hangers, people or watermarks.”

둘째 시트: “Premium square catalogue contact sheet, photorealistic garment studio flat-lay, precisely 3 columns x 3 rows. Soft diffused light, pale warm grey background. Row 1: light blue and white vertical striped Oxford shirt, white Breton T-shirt with navy horizontal stripes, olive and cream gingham checked shirt. Row 2: beige crewneck knit sweater, cream cable-knit sweater, navy polo shirt. Row 3: grey tailored trousers, olive chinos, off-white cream jeans. Detailed cotton fibers, wool texture, seam stitching, buttons and natural wrinkles. Full garments visible. No text, UI, people or watermarks.”

등록 사진: “A single complete light blue cotton Oxford button-down shirt, front-facing flat lay for a wardrobe app photo-registration example. Photorealistic studio photograph, woven cotton, long sleeves, collar, chest pocket, white buttons, stitching and subtle wrinkles. Solid light blue, no stripes or checks. Pale warm grey background, soft daylight, full garment visible with safe margins. No model, hanger, brand marks, UI or watermarks. Clearly identified AI-generated demonstration photograph, not an actual commercial product.”

실행 시에는 위 구성에 정렬·색상·전체 의류가 잘리지 않도록 하는 제약을 추가하였다. 이미지 파일 세 개를 프로젝트 안에 복사했고, 사진 속성 확인 예시는 실제 NVIDIA API 반환을 별도로 기록했다.
