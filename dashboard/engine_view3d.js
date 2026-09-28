/**
 * ENGINE-TWIN: Realistic 3D Digital Twin Viewer
 *
 * Loads a representative inline-4 turbo aero-diesel GLB instead of the old
 * block/procedural engine. The GLB is deliberately generic/representative;
 * telemetry, health and fault logic still come from the existing twin.
 */

class Engine3DView {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    this.engineRoot = null;
    this.modelLoaded = false;
    this.cylinders = [];
    this.propeller = null;
    this.turboMesh = null;
    this.turboParts = [];
    this.oilSumpMesh = null;
    this.oilSumpParts = [];
    this.statusLabel = null;
    this.selectedComponent = null;
    this.selectedMeshes = [];
    this.viewMode = "nominal";
    this.baseModelScale = 1.0;
    this.flowPulse = 0;

    this.currentRpm = 1400.0;
    this.targetCameraPos = null;
    this.targetLookAt = null;

    // Dashboard-served GLB. Drop a different representative model at this
    // path later without changing the viewer logic.
    this.modelUrl = '/static/assets/engine/VRDE_2.2L_Realistic_Inline4.glb';

    this.init();
  }

  init() {
    const width = this.container.clientWidth || 400;
    const height = this.container.clientHeight || 300;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090d16);

    this.camera = new THREE.PerspectiveCamera(42, width / height, 0.05, 100);
    this.camera.position.set(5.2, 3.2, 6.4);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    if ('outputEncoding' in this.renderer) this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    if (window.THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxDistance = 16;
      this.controls.minDistance = 1.5;
      this.controls.target.set(0, 0.25, 0);
    }

    // Engine-showcase lighting: lower ambient fill + controlled key/rim lights
    // so aluminum, painted metal, exhaust and dark rubber remain visually distinct.
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    const hemi = new THREE.HemisphereLight(0xb9d7ee, 0x05070b, 0.52);
    this.scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.05);
    keyLight.position.set(5.5, 7.5, 6.5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    this.scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x9db7d1, 0.42);
    fillLight.position.set(-5, 2.5, 4.5);
    this.scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x67d5ff, 0.62);
    rimLight.position.set(-5, 4.5, -6.5);
    this.scene.add(rimLight);

    const warmLight = new THREE.DirectionalLight(0xffd8ad, 0.22);
    warmLight.position.set(4, -2, 5);
    this.scene.add(warmLight);

    const grid = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    grid.position.y = -1.65;
    this.scene.add(grid);

    this.createStatusLabel();
    this.loadEngineModel();

    this.renderer.domElement.addEventListener('pointerdown', (event) => this.handlePointerSelect(event));
    window.addEventListener('resize', () => this.onWindowResize());
    this.animate();
  }

  createStatusLabel() {
    this.statusLabel = document.createElement('div');
    this.statusLabel.style.position = 'absolute';
    this.statusLabel.style.left = '10px';
    this.statusLabel.style.bottom = '10px';
    this.statusLabel.style.padding = '4px 7px';
    this.statusLabel.style.border = '1px solid rgba(58,56,48,.8)';
    this.statusLabel.style.borderRadius = '0';
    this.statusLabel.style.background = 'rgba(16,18,16,.85)';
    this.statusLabel.style.color = '#a8a6a0';
    this.statusLabel.style.font = '10px JetBrains Mono, monospace';
    this.statusLabel.style.pointerEvents = 'none';
    this.statusLabel.innerText = 'LOADING 3D ENGINE MODEL...';
    this.container.style.position = 'relative';
    this.container.appendChild(this.statusLabel);
  }

  loadEngineModel() {
    if (!window.THREE.GLTFLoader) {
      this.statusLabel.innerText = 'GLTF LOADER NOT AVAILABLE';
      this.statusLabel.style.color = '#c07670';
      return;
    }

    const loader = new THREE.GLTFLoader();
    loader.load(
      this.modelUrl,
      (gltf) => {
        this.engineRoot = gltf.scene;
        this.engineRoot.name = 'VRDE_2.2L_Representative_Engine';

        // Make the imported CAD-style materials behave well in the dark GCS UI.
        this.engineRoot.traverse((obj) => {
          if (!obj.isMesh) return;
          obj.castShadow = true;
          obj.receiveShadow = true;
          if (obj.material) {
            if (Array.isArray(obj.material)) obj.material = obj.material.map(m => this.prepareMaterial(m));
            else obj.material = this.prepareMaterial(obj.material);
            this.styleImportedPart(obj);
          }
        });

        this.scene.add(this.engineRoot);
        this.fitModelToViewport();
        this.indexComponents();
        this.modelLoaded = true;
        this.statusLabel.innerText = 'VRDE 2.2L INLINE-4 • REALISTIC 3D DIGITAL TWIN';
        this.statusLabel.style.color = '#9aa876';
        this.statusLabel.style.borderColor = 'rgba(95,106,69,.8)';
      },
      undefined,
      (error) => {
        console.error('3D engine GLB load failed:', error);
        this.statusLabel.innerText = '3D MODEL LOAD FAILED — CHECK /static/assets/engine/';
        this.statusLabel.style.color = '#c07670';
      }
    );
  }

  prepareMaterial(material) {
    const m = material.clone ? material.clone() : material;

    // Preserve imported maps, but give the CAD asset a restrained engineering
    // material palette instead of letting the strong dashboard lights wash it out.
    if (m.color && m.color.r + m.color.g + m.color.b < 0.08) {
      m.color.setRGB(0.07, 0.08, 0.09);
    }
    if ('metalness' in m) m.metalness = Math.min(0.90, Math.max(0.35, m.metalness || 0.60));
    if ('roughness' in m) m.roughness = Math.min(0.72, Math.max(0.22, m.roughness || 0.34));
    return m;
  }

  styleImportedPart(obj) {
    if (!obj || !obj.isMesh || !obj.material) return;
    const name = (obj.name || '').toLowerCase();

    const apply = (m) => {
      if (!m || !m.color) return;

      // Only recolor untextured materials; textures remain untouched.
      if (!m.map) {
        if (name.includes('oil') || name.includes('belt') || name.includes('dipstick')) {
          m.color.setHex(0x20272b);
          if ('roughness' in m) m.roughness = 0.48;
        } else if (name.includes('exhaust') || name.includes('turbine') || name.includes('wastegate')) {
          m.color.setHex(0x684734);
          if ('metalness' in m) m.metalness = 0.82;
          if ('roughness' in m) m.roughness = 0.34;
        } else if (name.includes('turbo') || name.includes('compressor')) {
          m.color.setHex(0x56636b);
          if ('metalness' in m) m.metalness = 0.90;
          if ('roughness' in m) m.roughness = 0.24;
        } else if (name.includes('fin')) {
          m.color.setHex(0x3f4b52);
          if ('metalness' in m) m.metalness = 0.78;
          if ('roughness' in m) m.roughness = 0.38;
        } else if (name.includes('rocker') || name.includes('head')) {
          m.color.setHex(0x68757d);
          if ('metalness' in m) m.metalness = 0.88;
          if ('roughness' in m) m.roughness = 0.27;
        } else if (name.includes('injector') || name.includes('bolt') || name.includes('rim') || name.includes('pulley')) {
          m.color.setHex(0xa5afb5);
          if ('metalness' in m) m.metalness = 0.92;
          if ('roughness' in m) m.roughness = 0.20;
        } else if (name.includes('propeller')) {
          m.color.setHex(0x2c3439);
          if ('metalness' in m) m.metalness = 0.55;
          if ('roughness' in m) m.roughness = 0.38;
        } else if (name.includes('crankcase') || name.includes('sump') || name.includes('flywheel')) {
          m.color.setHex(0x344149);
          if ('metalness' in m) m.metalness = 0.84;
          if ('roughness' in m) m.roughness = 0.32;
        }
      }
    };

    if (Array.isArray(obj.material)) obj.material.forEach(apply);
    else apply(obj.material);
  }

  fitModelToViewport() {
    if (!this.engineRoot) return;

    const box = new THREE.Box3().setFromObject(this.engineRoot);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Normalize any replacement model to roughly the same visual footprint.
    const targetSize = 4.2;
    const scale = targetSize / Math.max(maxDim, 0.001);
    this.engineRoot.scale.multiplyScalar(scale);

    const box2 = new THREE.Box3().setFromObject(this.engineRoot);
    const center2 = box2.getCenter(new THREE.Vector3());
    this.engineRoot.position.sub(center2);
    this.engineRoot.position.y += 0.05;

    // Slight presentation yaw exposes the turbo/exhaust side while retaining
    // the inline-four silhouette. Orbit mode can still rotate freely.
    this.engineRoot.rotation.y = 0.08;

    // Closer framing so the model reads as the dashboard centerpiece.
    this.camera.position.set(2.29, 2.36, 5.66);
    if (this.controls) this.controls.target.set(0, 0.2, 0);
  }

  indexComponents() {
    this.cylinders = [];
    this.turboParts = [];
    this.oilSumpParts = [];
    this.propeller = new THREE.Group();
    this.propeller.name = 'PropellerControlGroup';

    if (!this.engineRoot) return;

    const cylinderNodes = [1, 2, 3, 4].map(i => []);
    const turboNodes = [];
    const sumpNodes = [];
    const propNodes = [];

    this.engineRoot.traverse((obj) => {
      if (!obj.isMesh) return;
      const n = (obj.name || '').toLowerCase();

      const cylMatch = n.match(/cylinder[_ -]?(1|2|3|4)|c(1|2|3|4)[_ -]/i);
      if (cylMatch) {
        const idx = Number(cylMatch[1] || cylMatch[2]) - 1;
        if (idx >= 0 && idx < 4) cylinderNodes[idx].push(obj);
      }

      if (n.includes('turbo')) turboNodes.push(obj);
      if (n.includes('oil_sump') || n.includes('oil sump') || n.includes('oil-filter')) sumpNodes.push(obj);
      if (n.includes('propeller') || n.includes('prop_')) propNodes.push(obj);
    });

    for (let i = 0; i < 4; i++) {
      const group = new THREE.Group();
      group.name = `CylinderControl_${i + 1}`;
      cylinderNodes[i].forEach(mesh => group.attach(mesh));
      this.engineRoot.add(group);
      this.cylinders.push({ group, meshes: cylinderNodes[i], index: i, baseMaterials: this.captureMaterials(cylinderNodes[i]) });
    }

    turboNodes.forEach(mesh => {
      this.turboParts.push(mesh);
      this.prepareHighlightMaterials(mesh);
    });
    sumpNodes.forEach(mesh => {
      this.oilSumpParts.push(mesh);
      this.prepareHighlightMaterials(mesh);
    });
    if (propNodes.length) {
      const propBox = new THREE.Box3();
      propNodes.forEach(mesh => propBox.expandByObject(mesh));
      const pivot = propBox.getCenter(new THREE.Vector3());
      this.engineRoot.add(this.propeller);
      this.propeller.position.copy(pivot);
      propNodes.forEach(mesh => this.propeller.attach(mesh));
    } else {
      this.engineRoot.add(this.propeller);
    }

    this.turboMesh = this.turboParts[0] || null;
    this.oilSumpMesh = this.oilSumpParts[0] || null;
  }

  captureMaterials(meshes) {
    const result = [];
    meshes.forEach(mesh => {
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      materials.forEach(m => {
        if (m && m.emissive) result.push({ material: m, color: m.emissive.clone(), intensity: m.emissiveIntensity || 0 });
      });
      this.prepareHighlightMaterials(mesh);
    });
    return result;
  }

  prepareHighlightMaterials(mesh) {
    if (!mesh.material) return;
    const cloneOne = (m) => {
      const c = m.clone ? m.clone() : m;
      if (c.emissive) {
        c.emissive = c.emissive.clone();
        c.emissiveIntensity = c.emissiveIntensity || 0;
      }
      return c;
    };
    if (Array.isArray(mesh.material)) mesh.material = mesh.material.map(cloneOne);
    else mesh.material = cloneOne(mesh.material);
  }

  setMeshGlow(mesh, hex, intensity) {
    if (!mesh || !mesh.material) return;
    const apply = (m) => {
      if (!m.emissive) return;
      m.emissive.setHex(hex);
      m.emissiveIntensity = intensity;
    };
    if (Array.isArray(mesh.material)) mesh.material.forEach(apply);
    else apply(mesh.material);
  }

  setGroupGlow(meshes, hex, intensity) {
    meshes.forEach(mesh => this.setMeshGlow(mesh, hex, intensity));
  }

  resetGroupGlow(meshes) {
    meshes.forEach(mesh => {
      if (!mesh.material) return;
      const apply = (m) => {
        if (m.emissive) {
          m.emissive.setHex(0x000000);
          m.emissiveIntensity = 0;
        }
      };
      if (Array.isArray(mesh.material)) mesh.material.forEach(apply);
      else apply(mesh.material);
    });
  }

  setCameraPreset(preset) {
    if (!this.camera || !this.controls) return;
    if (preset === 'orbit') {
      this.targetCameraPos = new THREE.Vector3(5.2, 3.0, 6.2);
      this.targetLookAt = new THREE.Vector3(0, 0.2, 0);
    } else if (preset === 'cylinders') {
      this.targetCameraPos = new THREE.Vector3(0.1, 3.2, 4.2);
      this.targetLookAt = new THREE.Vector3(0, 0.65, 0);
    } else if (preset === 'turbo') {
      this.targetCameraPos = new THREE.Vector3(-4.0, 1.7, 3.0);
      this.targetLookAt = new THREE.Vector3(-1.2, 0.6, -0.55);
    } else if (preset === 'sump') {
      this.targetCameraPos = new THREE.Vector3(3.2, -2.0, 4.0);
      this.targetLookAt = new THREE.Vector3(0, -0.75, 0);
    }
  }

  focusCylinder(index) {
    if (index >= 0 && index < 4 && this.cylinders[index]) {
      const box = new THREE.Box3().setFromObject(this.cylinders[index].group);
      const center = box.getCenter(new THREE.Vector3());
      this.targetCameraPos = new THREE.Vector3(center.x + 1.1, center.y + 1.2, center.z + 2.5);
      this.targetLookAt = center;
    }
  }

  clearComponentSelection() {
    this.selectedMeshes.forEach(mesh => {
      if (!mesh || !mesh.material) return;
      const apply = (m) => {
        if (m && m.emissive) { m.emissive.setHex(0x000000); m.emissiveIntensity = 0; }
      };
      if (Array.isArray(mesh.material)) mesh.material.forEach(apply); else apply(mesh.material);
    });
    this.selectedMeshes = [];
    this.selectedComponent = null;
  }

  selectComponent(key) {
    if (!this.engineRoot) return false;
    this.clearComponentSelection();
    const lower = String(key || '').toLowerCase();
    let meshes = [];
    let label = key;
    let target = new THREE.Vector3(0, 0.2, 0);

    const add = (mesh) => { if (mesh && mesh.isMesh && !meshes.includes(mesh)) meshes.push(mesh); };
    if (/^cylinder[1-4]$/.test(lower)) {
      const idx = Number(lower.slice(-1)) - 1;
      if (this.cylinders[idx]) {
        this.cylinders[idx].meshes.forEach(add);
        label = `CYLINDER 0${idx + 1}`;
        const box = new THREE.Box3().setFromObject(this.cylinders[idx].group);
        target.copy(box.getCenter(new THREE.Vector3()));
      }
    } else {
      const patterns = {
        piston: ['piston'], crankshaft: ['crankshaft','crank_'], valvetrain:['valve','rocker','camshaft'],
        intake:['intake'], exhaust:['exhaust'], turbo:['turbo']
      };
      (patterns[lower] || [lower]).forEach(pattern => {
        this.engineRoot.traverse(obj => { if (obj.isMesh && (obj.name || '').toLowerCase().includes(pattern)) add(obj); });
      });
      if (meshes.length) {
        const box = new THREE.Box3(); meshes.forEach(m => box.expandByObject(m)); target.copy(box.getCenter(new THREE.Vector3()));
        label = lower.toUpperCase().replace('VALVETRAIN','VALVE TRAIN');
      }
    }
    if (!meshes.length) return false;
    this.selectedMeshes = meshes;
    this.selectedComponent = label;
    this.setGroupGlow(meshes, 0x50dfff, 1.15);
    const box = new THREE.Box3(); meshes.forEach(m => box.expandByObject(m));
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.8);
    const dist = Math.max(2.6, Math.min(6.0, maxDim * 3.6));
    this.targetCameraPos = new THREE.Vector3(target.x + dist * 0.55, target.y + dist * 0.35, target.z + dist);
    this.targetLookAt = target;
    return true;
  }

  setViewMode(mode) {
    if (!this.engineRoot) return;
    this.viewMode = mode || 'nominal';
    // Presentation-only modes. Geometry and telemetry logic are untouched.
    this.engineRoot.visible = true;
    this.engineRoot.traverse(obj => {
      if (!obj.isMesh || !obj.material) return;
      const apply = (m) => {
        if (!m) return;
        if ('transparent' in m) m.transparent = false;
        if ('opacity' in m) m.opacity = 1;
      };
      if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
    });
    if (mode === 'exploded') {
      const offsets = {1:-0.22,2:-0.07,3:0.08,4:0.23};
      this.cylinders.forEach((c,i)=>{ if(c && c.group) c.group.position.z = offsets[i] || 0; });
      if (this.turboParts.length) this.turboParts.forEach(m=>{m.position.x += 0.15;});
    } else {
      this.cylinders.forEach(c=>{ if(c && c.group) c.group.position.z = 0; });
    }
    if (mode === 'cutaway') {
      this.engineRoot.traverse(obj => {
        if (!obj.isMesh || !obj.material) return;
        const apply = (m) => { if (m && !m.map) { m.transparent = true; m.opacity = 0.34; } };
        if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
      });
    }
    if (mode === 'thermal') {
      this.engineRoot.traverse(obj => {
        if (!obj.isMesh || !obj.material) return;
        const apply = (m) => { if (m && !m.map && m.color) { m.color.setHex(0x384d45); if(m.emissive){m.emissive.setHex(0x6b351f);m.emissiveIntensity=.35;} } };
        if (Array.isArray(obj.material)) obj.material.forEach(apply); else apply(obj.material);
      });
    }
    if (mode === 'airflow') {
      this.setGroupGlow(this.turboParts, 0x50dfff, 0.85);
      this.setGroupGlow(this.oilSumpParts, 0x50dfff, 0.25);
    }
    if (mode === 'structural') {
      this.setGroupGlow(this.cylinders.flatMap(c=>c.meshes), 0xf0c20a, 0.45);
      this.setGroupGlow(this.turboParts, 0xff3e48, 0.35);
    }
  }

  updateFromTwinState(state) {
    if (!state) return;
    this.currentRpm = state.sensor_rpm || 1400.0;

    const cylHealths = (state.health && state.health.cylinder_health) || [100, 100, 100, 100];
    const chtTemps = state.sensor_cht_c || [175, 175, 175, 175];

    for (let i = 0; i < 4; i++) {
      if (!this.cylinders[i]) continue;
      const h = Number(cylHealths[i] ?? 100);
      const t = Number(chtTemps[i] ?? 175);

      let glowColor = 0x000000;
      let intensity = 0.0;
      if (h < 50.0 || t > 215.0) {
        glowColor = 0xdc2626;
        intensity = 0.95;
      } else if (h < 80.0 || t > 190.0) {
        glowColor = 0xd97706;
        intensity = 0.55;
      }
      this.setGroupGlow(this.cylinders[i].meshes, glowColor, intensity);
    }

    const oilHealth = Number(state.health?.oil_system_health ?? 100);
    this.setGroupGlow(this.oilSumpParts, oilHealth < 50 ? 0xb91c1c : 0x000000, oilHealth < 50 ? 0.75 : 0);

    const turboHealth = Number(state.health?.turbo_boost_health ?? 100);
    this.setGroupGlow(this.turboParts, turboHealth < 60 ? 0xd97706 : 0x000000, turboHealth < 60 ? 0.80 : 0.0);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.propeller && this.propeller.children.length) {
      const rotSpeed = (this.currentRpm / 4000.0) * 0.40 + 0.05;
      this.propeller.rotation.x += rotSpeed;
    }
    if (this.viewMode === 'airflow') {
      this.flowPulse += 0.045;
      const pulse = (Math.sin(this.flowPulse) + 1) * 0.5;
      this.setGroupGlow(this.turboParts, 0x50dfff, 0.35 + pulse * 0.75);
    }

    if (this.targetCameraPos && this.camera && this.controls) {
      this.camera.position.lerp(this.targetCameraPos, 0.08);
      this.controls.target.lerp(this.targetLookAt, 0.08);
      if (this.camera.position.distanceTo(this.targetCameraPos) < 0.04) {
        this.targetCameraPos = null;
        this.targetLookAt = null;
      }
    }

    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
  }

  handlePointerSelect(event) {
    if (!this.engineRoot || !this.camera || !this.renderer) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(pointer, this.camera);
    const hits = raycaster.intersectObject(this.engineRoot, true);
    if (!hits.length) return;
    const n = (hits[0].object.name || '').toLowerCase();
    let key = null;
    const cyl = n.match(/c([1-4])_|cylinder[_ -]?([1-4])/i);
    if (cyl) key = `cylinder${Number(cyl[1] || cyl[2])}`;
    else if (n.includes('piston')) key='piston';
    else if (n.includes('crank')) key='crankshaft';
    else if (n.includes('rocker') || n.includes('valve') || n.includes('cam')) key='valvetrain';
    else if (n.includes('intake')) key='intake';
    else if (n.includes('exhaust')) key='exhaust';
    else if (n.includes('turbo')) key='turbo';
    if (key && window.handle3DComponentSelection) window.handle3DComponentSelection(key);
  }

  onWindowResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = Math.floor(this.container.clientWidth);
    const height = Math.floor(this.container.clientHeight);
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }
}
