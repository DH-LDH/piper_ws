# MIT 토크제어 + 관절 SMC 진행 체크리스트

근거: 2026-10-10 박사님 피드백(MIT 토크제어 → 로봇팔 SMC 논문 구현 → 결과 보고 강화학습 여부 판단),
SS-ASMC 논문 리뷰(Baek & Kwon 2020)의 적용 순서, [MIT_E2_LOG.md](MIT_E2_LOG.md) 4절.

## 완료

- [x] 궤적·IK 직접 계산(`joint_ik`)으로 끊김 제거 — [B_STAGE_INTERIM.md](B_STAGE_INTERIM.md)
- [x] EE 그래프 x/y/z 축별 + 3D, 좌표계(base_link) 명시 — 10-10 피드백
- [x] 펌웨어 MOVE L 결과 xyz 분해 — A단계 place_descend
- [x] MIT 진입 절차(Enable·모드 전환 분리) — 진입 튐 없음 (E2)

## 0. 장비 복구 (사람 손 필요)

- [ ] 팔 전원 켜기
- [ ] USB-CAN 어댑터 뽑았다 다시 꽂기 → `can_piper` 이름 확인
- [ ] `sudo ip link set can_piper down && sudo ip link set can_piper type can bitrate 1000000 && sudo ip link set can_piper up`
- [ ] 피드백 200 Hz 확인(`GetArmJointMsgs().Hz`), `candump can_piper` 수신 확인

## 1. E3 — 순간 튐 가설 판정 (`--ext-kp`)

- [ ] `clear_arm_fault.py` → `traj_test` 직선으로 시작 자세 이동
- [ ] `mit_probe.py --stage hold --hold 15 --go --ext-kp 10 10 10 10 10 10`
- [ ] 판정: 15 s 동안 튐 0회면 "펌웨어가 목표 위치를 0으로 처리" 가설 확정 → 이후 모든 제어는 펌웨어 Kp = 0, τ_ff로만
- [ ] 튐이 남으면 가설 기각 → Kd도 0으로 두고 재시험, 그래도 남으면 원인 재조사
- [ ] 결과를 MIT_E2_LOG.md에 추가(또는 MIT_E3_LOG.md)

## 2. 한 축 TDC (TDE + 극점 배치만)

- [ ] 대상 관절 하나(J6 또는 J5), 나머지는 E3 구조로 유지
- [ ] q̈: 속도 피드백 미분 + 저역통과 필터, 필터 지연 기록
- [ ] τ_t = τ_{t−L} − M̄·q̈_{t−L} + M̄(q̈_d + (Ks+Kg)ė + Ks·Kg·e), M̄은 아주 작게 시작해서 키움
- [ ] τ_{t−L}은 보낸 τ_ff 사용(Kd = 0일 때 일치) — 토크 피드백 사용과 비교
- [ ] 루프 100 Hz(mit_probe 기준), 매 주기 τ, e, s, TDE 오차 E 기록
- [ ] 정지 유지 → 사인파 추종 순서로, 위치 모드 대비 오차·떨림 비교

## 3. 한 축 TDC + 고정 게인 스위칭

- [ ] τ에 M̄·K·sgn(s) 추가(먼저 경계층 포화형), K 몇 단계로
- [ ] τ 그래프로 채터링 직접 확인(논문에 없는 부분)

## 4. 한 축 SS-ASMC (parent·child 적응)

- [ ] 식 (13) parent, 식 (14) child 구현, dK̂/dt는 ±μ로 포화
- [ ] K̂, η̂, s, τ 기록 — 2·3단계와 RMS 오차·채터링 비교

## 5. 6축 확장 + 궤적 추종

- [ ] 6축 대각 독립 적용, χ는 ‖s‖∞
- [ ] `joint_ik` 느린 직선을 기준 신호로 → B단계 위치 모드(상대 진동 0.61/0.60)와 같은 지표로 비교
- [ ] EE x/y/z + 3D 그래프, 좌표계·속도 계산법 명시
- [ ] 별도 중력 보상 필요 여부 확인(TDE가 중력도 상쇄하는지)

## 6. 보고·판단

- [ ] 결과 정리 후 박사님 보고 — 강화학습으로 넘어갈지 판단 근거(남은 떨림 크기, 튜닝 부담)
